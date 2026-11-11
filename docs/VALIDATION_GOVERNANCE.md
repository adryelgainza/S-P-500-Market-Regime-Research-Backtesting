# Frozen validation and experiment governance

## Implemented contracts

Layer 5 provides a versioned scientific protocol, source and environment archives, an append-only experiment register, chronological validation, fixed-weight transfer aggregation, and separate methodology and strategy-evidence gates.

The supported commands run offline. The initial release imports the saved historical study and freezes a subsequent evaluation design. Transfer and forward performance await their registered observations and audit-backed evaluations.

| Component | Responsibility |
| --- | --- |
| `protocol.py` | Validate commitments, archive scientific source and dependency pins, freeze with the actual clock, verify receipts |
| `experiments.py` | Serialize writers, append hash-linked events, preserve failures and corrections, save trusted heads |
| `validation.py` | Check training and cash chronology, rebuild raw prefixes, bind stages and recipes, compute and archive evidence |
| `validation_research.py` | Import historical evidence, publish comparison tables, report status, verify saved inputs and gates |

## Initial scientific commitment

The protocol is `protocols/layer5_validation_v1.json`, identified as `downside-overlay`, version `1.0.0`. These choices were made during implementation after the historical findings had been inspected. Their motivation is a simple, reviewable test of the existing downside-risk hypothesis.

| Commitment | Frozen setting |
| --- | --- |
| Primary instrument | SPY |
| Fixed model | `HGB_Downside_10d_W20_Full_A` |
| Allocation | `AlwaysLong_Step_P90`, long/cash, next-open execution |
| Model updates | Initial eligible fit and annual refits, 20-year training window, seed 42 |
| Selection | One fixed candidate, benchmark selection ineligible |
| Primary comparator | Identical-ledger buy-and-hold |
| Additional comparator | SMA 50/200 |
| Transfer instruments | QQQ and IEF, predetermined 50/50 daily capital weights |
| Transfer window | 2024-01-02 through 2026-07-23, all eligible registered sessions required |
| Primary cost | 5 bps one-way on actual traded notional |
| Cost sensitivities | 0, 5, 10 and 20 bps |
| Economic minimum | Delta-Sharpe at least 0.10 |
| Uncertainty | 95% paired circular-block interval, 20-session blocks, 10,000 draws, lower bound above zero |
| Mean-return test | One-sided block max-T, 2,000 draws, HAC lag 20, adjusted p at most 0.05 |
| DSR | At least 0.95 for the single fixed fresh-evidence family |
| Risk | Maximum drawdown magnitude at most 35% |
| Cost gate | Nonnegative delta-Sharpe at 10 bps one-way |
| Review horizon | First 756 completed scheduled forward return intervals |
| Episode minimum | 10 invested episodes, starting on a zero-to-positive target transition |

Model parameters, the full feature list, calibration, score ranking, hysteresis, target definition and source implementations are archived. The target describes raw executable-price downside risk. Portfolio returns include the registered action and cash conventions.

The 756-session and 10-episode requirements can produce an inconclusive result even if point estimates are positive. The observation count is a fixed administrative review horizon. An extension creates a new protocol and forward window. Monitoring occurs monthly and descriptive reporting annually. Early performance-based promotion conflicts with the protocol.

QQQ and IEF have no recorded result inspection in this project. Their transfer histories still share calendar time with the development data. Their inspection flags and per-market results must be retained. Each instrument must meet the history and score warmup requirements. Missing eligibility makes the complete transfer release incomplete.

## Evidence stages

```text
Inspected history -> retrospective development
                         |
                         v
              actual protocol freeze
                   /           \
                  v             v
       registered transfer   future decisions
       fixed markets/dates   committed before execution
                  |             |
                  v             v
          transfer report    fixed-horizon forward report
```

Historical import timestamps describe the import. Original run IDs, provenance and completion timestamps remain in each imported specification. The original 182-candidate statistical family is preserved. Layer 3's max-T family includes the reported selector and therefore has 183 tested streams. Four earlier locked model families, accounting replays and randomized controls are separately disclosed. The broader project search count remains unknown because earlier informal attempts are incompletely reconstructable.

The freeze receipt records the first exchange close plus one minute strictly after the actual freeze timestamp. Execution occurs at the following exchange open. Eligibility also requires current data, training history and score warmup. A missing scheduled forward observation makes the complete fixed-window evaluation incomplete. Stage validation rejects historical rows assigned to forward evaluation and requires a contemporaneously recorded target for every forward decision.

## Commands

Run from the project directory using the project environment.

```powershell
python -m regime_research.validation_research import-history
python -m regime_research.validation_research freeze protocols/layer5_validation_v1.json
python -m regime_research.validation_research report --frozen experiments/frozen/downside-overlay/1.0.0 --output runs/layer5_validation_governance
python -m regime_research.validation_research verify --report runs/layer5_validation_governance
python -m regime_research.validation_research verify --frozen experiments/frozen/downside-overlay/1.0.0
python -m regime_research.validation_research verify
```

Imports are idempotent. An identical freeze retry returns its existing receipt. Editing a frozen version is rejected. Assign a new version for changed commitments or scientific source. Reports use a fresh output directory. The report verifier reconstructs the experiment table from its anchored log prefix, checks artifact bytes, and recomputes registered evaluation metrics and gates.

Freeze directories contain the protocol, exact installed package pins, Python/platform metadata, scientific source, notebook code, dataset manifests, and the receipt. The source revision is a SHA-256-addressed archive. Layer 6 adds the Git baseline, supported dependency lock and strict fresh-kernel runner. Notebook explanatory Markdown is excluded from the scientific source digest. Version 1 remains archived. Version 2 binds the extracted implementation and starts its own actual forward window.

## Registering an attempt

Each attempt has its own experiment ID. Repeated runs retain the same hypothesis ID and link to their parent attempt. Registration records the scientific configuration hash, protocol hash, source revision, exact dataset IDs, stage, seed and purpose.

```python
from regime_research.experiments import ExperimentStore, content_hash
from regime_research.protocol import verify_freeze

store = ExperimentStore(project / "experiments")
frozen = project / "experiments/frozen/downside-overlay/1.0.0"
verified = verify_freeze(frozen, store, project=project)
protocol = verified["Protocol"]
body = verified["Receipt"]["Body"]
specification = {
    "Hypothesis_ID": protocol["Hypothesis"]["ID"],
    "Protocol_SHA256": body["Protocol_SHA256"],
    "Configuration_SHA256": content_hash({
        key: protocol[key] for key in ["Models", "Allocation"]
    }),
    "Source_Revision": body["Source_Revision"],
    "Dataset_IDs": [exact_dataset_id],
    "Parent_Experiment_ID": previous_attempt_id,
    "Stage": "transfer",
    "Seed": protocol["Models"]["Seed"],
    "Purpose": "registered transfer evaluation",
}
with store.attempt(specification) as (experiment_id, outcome):
    # Run the frozen rule and the required independent audit functions here.
    # Save artifact references with project-relative paths and SHA-256 digests.
    outcome["Artifacts"] = artifact_references
    outcome["Result"] = result_summary
```

The context records registration and start before work begins, completion on success, and failure on an exception or keyboard interruption. A process killed without cleanup retains a visible started attempt. On recovery, append a failed or abandoned event with the reason. Register its replacement with a new ID and the same hypothesis. The original record stays reviewable.

Corrections append `experiment_corrected` with `Corrects_Event_ID`, a reason and replacement information. They must target an event from the same experiment. Comparison tables retain the original fields and show the correction details alongside them.

Forward attempts use the `forward` stage. `record_forward_decision` requires the frozen configuration and source, a current immutable snapshot, hashed score/audit artifacts, a scheduled decision timestamp, a next-open execution timestamp and a weight in [0, 1]. It accepts the signal only after its decision time and before execution. Retries preserve the first committed signal. The CLI provides `record-decision` for an already started attempt.

## Chronology and evaluation APIs

| Function | Required check |
| --- | --- |
| `validate_refit` | Unique base/calibration row IDs, feature availability by decision, base labels strictly before calibration, calibration labels strictly before refit |
| `validate_selection` | Every selected return ends strictly before the selection decision |
| `validate_cash_timing` | Rate availability at or before the accrual interval starts |
| `check_prefix_invariance` | Rebuild decisions from raw data prefixes through the normal decision callable and compare with the complete-input result |
| `validate_run_binding` | The same frozen hypothesis, model/allocation hash, source, seed, protocol and registered instrument |
| `validate_stage` | Valid execution timeline, realized returns, complete registered transfer windows and committed forward targets |
| `aggregate_transfer` | Every declared market, fixed weights, identical dates and strategy columns, finite returns |
| `evaluate_fixed_returns` | Identical dated net/excess streams at primary and gate costs, bounded episode weights, fixed horizon, registered inference |
| `assess_gates` | Separate research-validity and strategy-evidence outcomes, with inadequate or undefined evidence inconclusive |
| `record_evaluation` | A started bound attempt, stage checks, saved return/decision inputs, audit references and an appended evaluation event |
| `verify_evaluation` | Artifact integrity and independent recomputation of metrics, gates and stage from the saved evaluation inputs |

Fit audits use `Row_ID`, `Decision_Time`, `Available_At` and `Label_End_Time`. Selection audits use `Return_End_Time`. Cash audits use `Available_At` and `Interval_Start`. All timestamps require timezones.

`record_evaluation` accepts audit-derived validity states for chronology, accounting, provenance, reproducibility, trial disclosure and statistical calibration. Passing states require saved audit references. Their scientific meaning comes from the corresponding independent verifiers and raw-prefix checks. Missing audits remain inconclusive. Layer 6 provides strict model/notebook run receipts and independent study verification. Forward promotion also requires its contemporaneous decision and complete scheduled-observation audits.

Return inputs use a unique chronological UTC return-end index with `Net_Return` and `Net_Excess_Return`. Target weights share that exact index. The primary and 10 bps control streams must follow the same instrument, action, cash, execution and accounting conventions. Registered cost/block sensitivities and the harder comparator are reported alongside the primary evidence. A negative strategy outcome can coexist with passed research-validity gates.

## Integrity and recovery

Events contain a unique ID, UTC recording time, consecutive sequence, previous-event hash, idempotency key, payload and event hash. A writer lock serializes transactions. Complete-file replacement preserves the prior byte prefix and avoids publishing a partly written JSON line. Windows sharing failures receive bounded retries.

Freeze snapshots are fully staged before their event commits. A retry recovers a committed staged snapshot after interrupted publication. The freeze event binds the receipt body and prior log head. Archived heads in `experiments/anchors/` and the report's `experiment_anchor.json` detect alteration or truncation relative to that saved prefix. Keep a trusted copy of those heads in release storage or source control. Local hashes and local timestamps depend on the integrity of the local machine and trusted copies.

The report's comparison table includes every registered attempt in its anchored prefix, including failed and abandoned attempts. Registered statistical family size and broader project history have separate fields.

## Verification

```powershell
python -m unittest tests.unit.test_validation -v
python -m unittest discover -s tests -v
python -m unittest test_research_integrity -v
python verify_sharpe_research.py
```

The focused tests cover changed frozen versions, interrupted publication, atomic batch rejection, concurrent writers, idempotency conflicts, trusted-head truncation, corrections, failed/repeated attempts, immature labels, late cash observations, boundary-crossing returns, raw-prefix leakage, stage relabeling, late forward recording, per-market tuning, missing transfer markets, undefined inference and saved-evaluation tampering.

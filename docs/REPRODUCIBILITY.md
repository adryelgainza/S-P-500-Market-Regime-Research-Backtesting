# Reproduce and audit the research

## Release inputs and environment

The supported release uses CPython 3.13.7 on Windows x86-64. `requirements.lock` pins the 75-package dependency closure for numerical analysis, data adapters and notebook execution. `pyproject.toml` installs the local package and command entry points. The environment verifier checks the interpreter and every active pin before a supported run. Extra unrelated installed packages do not enter the scientific runtime contract.

The immutable historical baseline contains the corrected notebook, original data, package source through Layer 5, all 182 candidate streams, portfolio ledgers, statistical calibration, attribution results and experiment history. Its archive has 648 files and a SHA-256 manifest. It is stored at `archives/layer5_baseline/baseline.zip`. Git tracks its manifest. Carry the identified ZIP alongside a checkout or release because large input/result snapshots are stored outside Git.

The complete workspace input/result snapshot is `archives/layer6_release/workspace_inputs_and_results.zip`, bound by `archives/layer6_release/manifest.json`. Extract this archive into its matching checkout to restore datasets, saved model outputs, run evidence and the archived version 1 freeze. It preserves the Git-tracked active source and version 2 freeze. Keep the companion baseline ZIP at its identified path for independent historical replay.

The local baseline Git commit is `5a71417bce326553bf1d9bac97f5e5883b4cd2a1`. Content checksums preserve exact scientific bytes across platforms. The repository attributes disable line-ending conversion so a checkout retains those bytes. The active version 2 freeze bundle is small enough to be committed with its protocol and anchor. The large historical ZIP travels alongside the checkout.

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m pip check
```

The first command should report 3.13.7. A release can also include a wheel directory generated with `pip download --only-binary=:all: -r requirements.lock`. Installing with `--no-index --find-links <wheel-directory>` then provides an offline environment installation.

## One supported replay path

```powershell
.\.venv\Scripts\python.exe -m regime_research.runner run --mode replay --output runs/release_replay
.\.venv\Scripts\python.exe -m regime_research.runner verify runs/release_replay
```

The replay verifies the baseline ZIP and every archived member, materializes it under `checkpoints/baseline/`, and invokes its historical verifiers against its own source. It checks the original Sharpe artifacts, the 28 accounting portfolios, 182 trial streams, inference, matched controls and archived freeze/log head. Historical scientific version pins remain enforced. The archived freeze's full legacy environment is retained as provenance, while its archived verification uses its immutable bytes and anchor.

The current notebook then presents the verified results through the package. Original legacy metrics are regression-tested from fixed archived reference functions. `python verify_sharpe_research.py` remains a compatibility entry point for the archived verification.

The replay reads identified local files. Explicit acquisition remains available through `python data.py refresh-spy`. A refreshed snapshot has a different dataset ID and is selected deliberately for subsequent work. A source snapshot or input checksum mismatch aborts the replay.

Expect tens of seconds to a few minutes for archive verification and notebook replay, depending on storage and local kernel startup. The initial materialization writes approximately 140 MB of uncompressed baseline inputs and results. Later runs verify and reuse the complete materialization.

## Fixed-recipe development refit

```powershell
.\.venv\Scripts\python.exe -m regime_research.runner run --mode fit --output runs/fixed_recipe_development
.\.venv\Scripts\python.exe -m regime_research.runner verify runs/fixed_recipe_development
```

This entry point rebuilds causal features, matured execution labels, rolling annual HGB fits, calibration, the fixed allocation state machine, next-open portfolios, cost sensitivities, predictive diagnostics and inference. It uses the exact investable SPY snapshot from the registered protocol, with explicit actions and cash assumptions. Completed recipe/refit score blocks are stored under `checkpoints/models/`.

The fixed configuration is defined by `RunConfig` and `protocols/layer6_validation_v2.json`. A custom run configuration can be supplied with `--config <project-relative-json>`. Unknown fields and unsupported values are rejected. Dates use an exclusive end. The actual evaluation starts only after real-history, feature, fit and score-rank warmups are satisfied. Short valid samples retain their metrics and explicitly inconclusive inference.

The common rebalance policy trades when the rule target changes. Between target changes, shares drift and distribution cash remains in the cash account. The buy-and-hold control retains its initial shares, so its realized exposure can fall as distributions accumulate. Its return convention is explicit in the protocol.

The extracted model API preserves the historical HGB/RF classifier and HGB regression recipe families. The supported fixed SPY study implements the registered annual binary-downside recipe. The original 182-candidate selector remains independently replayable from the baseline. Rebuilding that entire historical search from cold fits is a separate expensive research job.

Version 2 uses explicit UTC after-close refits. A label that matured at that morning's exit open is eligible if it precedes the refit. The archived implementation used date-bound refits. Their results retain separate stage and source identities. New inspected-history refits are developmental. Fresh forward evaluation uses the actual new freeze time and signal commitments.

## Scientific cache and interruption contract

```text
recipe + fit configuration + seed
            |
            v
training values/labels + prediction features + timestamps
            |
            v
numerical runtime pins + scientific dependency hashes
            |
            v
per-refit score block + purge audit + atomic completion receipt
```

Cache keys hash feature names, dtypes, values, row order, matured labels, recipe, fit boundaries, seed, numerical library versions and scientific source dependencies. Rendering functions and explanatory Markdown are outside the predictor dependency set. Scientific changes or changed data invalidate affected checkpoints. Every run still binds the full current source and notebook cell plan.

A checkpoint has a score table, row-level base/calibration audit and checksummed completion file. Its completion file is published last. Incomplete, corrupted or incompatible blocks are rebuilt. Concurrent writers serialize publication. An interrupted fit retains complete earlier blocks. `refit_progress.jsonl` records recipe, refit time, score count and reuse status during the run.

Run directories are unique. Failed attempts keep `failure.json`, cell progress, kernel output and their event-log failure. A process killed before cleanup leaves a visible started attempt. Mark its disposition with a new failure/abandonment event and link its replacement to the same hypothesis. Completed research manifests are published after independent verification succeeds.

## Strict notebook execution

The runner launches the exact calling interpreter as a private Jupyter kernel with a fresh project-local IPython profile. It supplies an immutable saved cell plan. The kernel enforces that plan before execution and records a hash-linked audit. Nested `run_cell` calls are detected. The host correlates each request, input notification and execution reply, checks contiguous execution counts and one kernel session, and verifies source stability before each cell and after shutdown.

The executed notebook, complete cell plan, request IDs, source hashes, kernel audit and execution receipt are saved in the run directory. Extra, reordered, nested, missing or failed cells invalidate completion. Adding a legitimate saved cell creates a different scientific notebook digest and run identity. Untracked source changes during execution fail verification.

The implementation uses Jupyter's parent-message correlation and separate execution notifications and replies. The strict contract covers the supported runner and package entry points. Arbitrary malicious modification of the local interpreter or receipt files remains outside that local trust boundary. [Jupyter messaging specification](https://jupyter-client.readthedocs.io/en/stable/messaging.html)

## Completed-run manifest

| Binding | Recorded evidence |
| --- | --- |
| Source | Git revision, initial working-tree status, content-addressed scientific source map |
| Configuration | Validated run settings, full protocol, protocol digest and freeze receipt when available |
| Inputs | Exact dataset IDs, baseline archive SHA-256 and verified snapshot manifests |
| Environment | Python, platform, exact dependency versions and lock digest |
| Execution | Saved code-cell IDs/hashes, requests, counts, one kernel session and kernel-audit head |
| Outputs | Checksums for executed notebook, tables, charts and underlying chart data, ledgers and audits |
| Research history | Experiment ID, hypothesis, stage, seed and anchored experiment-log prefix |
| Verification | Independent accounting, return/metric reconstruction and study verification |

`pending_completion.json` is checked before `completion.json` is published. Failed or interrupted work retains its incomplete status. The independent verifier checks source, environment, inputs, artifact bytes, exact executed-cell sequence and the saved study again.

Predictor reuse and execution reuse have different scopes. A Markdown-only edit permits the same scientific predictor cache, while each fresh-kernel run still gets a newly observed receipt. Numerical comparisons use registered relative tolerance 1e-10 and absolute tolerance 1e-12. Source/configuration and saved-artifact checksums remain exact within a run.

## Verification tiers

| Tier | Command or procedure |
| --- | --- |
| Fast offline tests | `python -m unittest discover -s tests/unit -v` and `python -m unittest test_research_integrity -v` |
| Fresh-kernel integration | `python -m unittest tests.integration.test_runner -v` |
| Registered calibration | `python -m regime_research.statistics_calibration --output runs/new_calibration` |
| Full preserved-result replay | `python -m regime_research.runner run --mode replay --output runs/new_replay` |
| Development refit | Cold and warm `--mode fit` runs with aligned saved scores, targets, ledgers and metrics compared |
| Forward append | Frozen-source checks, current snapshot, timestamped signal commitments, chronology audits and complete scheduled observations |

The CI workflow runs fast tests and real small kernels on Windows with the same lock. Its manually dispatched calibration option runs the registered Monte Carlo suite. Full historical replay requires the release ZIP. CI's fixture-based checks run from a checkout without that large archive.

## Governance and current evidence

The version 1 freeze and its results remain archived. `validation_research verify --report runs/layer5_validation_governance --archived` checks the old source-bound receipt after the active implementation changes. New reports use portable project-relative freeze paths. The version 2 freeze records its own actual UTC time and starts a new forward window.

Validity attestations used for forward promotion must be backed by the corresponding fit, prefix, accounting, provenance and calibration audits, plus the strict supported-runner receipt. Saved-result replay establishes reproducibility of the historical result. It supplies zero fresh forward observations. Transfer studies and the registered forward observation horizon remain empirical work after this engineering release.

## Verified release outcomes

The clean Windows CPython 3.13.7 environment installed all 75 pinned dependencies and the local package. All 141 tests passed, including real fresh-kernel execution and raw-prefix model rebuilding. All five registered Monte Carlo scenarios passed their acceptance criteria. The historical notebook replay also passed from a separate clean Git checkout with the identified baseline ZIP.

The complete SPY cold run fitted 14 annual blocks. The warm run reused all 14. Their 71 scientific tables agree within the registered tolerances, including scores, targets, feature/label matrices, all 12 portfolio-cost ledgers and inference. The development 5 bps Sharpe is approximately 0.403 for the fixed overlay, 0.708 for the buy-and-hold cash-distribution convention and 0.527 for SMA 50/200. These results retain their development stage.

`runs/layer6_release/verification.json` contains the checks, exact run IDs, hashes, frozen timestamp and numerical comparisons. Each release run also has its own independently verifiable completed manifest and executed notebook. `archives/layer6_release/verification.json` provides the small Git-tracked summary alongside the large result snapshot's manifest.

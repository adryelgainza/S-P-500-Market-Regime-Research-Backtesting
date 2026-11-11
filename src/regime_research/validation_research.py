"""Offline governance commands for historical import, protocol freeze, and reporting."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .accounting_replay import verify_accounting_replay
from .attribution_research import verify_attribution_research
from .experiments import ExperimentStore, atomic_write, canonical_bytes, content_hash, utc_now
from .protocol import freeze_protocol, verify_freeze
from .statistical_research import verify_statistical_research
from .statistics import annualized_sharpe
from .trial_returns import load_trial_family
from .validation import assess_gates, record_forward_decision, verify_evaluation


HISTORY_ID = "historical-sharpe-v1.1"


def _artifact(project, path):
    path = Path(path)
    if not path.is_absolute():
        path = project/path
    return {"Path": path.relative_to(project).as_posix(), "SHA256": hashlib.sha256(path.read_bytes()).hexdigest()}


def import_history(project, store):
    """Import existing evidence with its original provenance and today's import time."""
    project = Path(project).resolve()
    verify_accounting_replay(project, project/"runs/layer2_accounting_replay")
    verify_statistical_research(project, project/"runs/layer3_statistical_inference")
    verify_attribution_research(project, project/"runs/layer4_attribution")
    _, _, trials, family_manifest = load_trial_family(project, project/"runs/layer3_candidate_family")
    registry = json.loads((project/"model_outputs/28_sharpe_experiment_registry.json").read_text())
    candidates = pd.read_csv(project/"model_outputs/32_sharpe_candidate_registry.csv").fillna("")
    if len(candidates) != 182 or set(candidates.Candidate_ID) != set(trials.columns):
        raise ValueError("Historical family differs from its declared 182 candidates")
    manifest = json.loads((project/"model_outputs/46_sharpe_run_manifest.json").read_text())
    dataset_id = family_manifest["Dataset_ID"]
    provenance = registry["Source_Provenance"]
    base = {"Hypothesis_ID": HISTORY_ID, "Configuration_SHA256": registry["Config_SHA256"],
            "Source_Revision": "historical-sha256-"+content_hash(provenance), "Dataset_IDs": [dataset_id],
            "Stage": "retrospective", "Seed": 42, "Purpose": "historical_model_search",
            "Original_Run_ID": manifest["Run_ID"], "Original_Completed_UTC": manifest["Completed_UTC"],
            "Source_Provenance": provenance}
    basis = "Retrospective import from saved source-bound artifacts. Registration occurred after outcomes were inspected."
    specifications = []
    def imported(identifier, specification, outcome):
        specifications.append({"Type": "experiment_imported", "Experiment_ID": identifier,
            "Idempotency_Key": "import-"+identifier, "Payload": {"Specification": specification,
            "Imported_State": "completed", "Import_Basis": basis, "Outcome": outcome}})
    imported(HISTORY_ID, base, {"Known_Candidate_Count": 182, "Artifacts": [
        _artifact(project, "model_outputs/46_sharpe_run_manifest.json"),
        _artifact(project, "runs/layer3_candidate_family/completion.json")]})
    for row in candidates.to_dict("records"):
        identifier = row["Candidate_ID"]
        specification = dict(base, Hypothesis_ID=f"{HISTORY_ID}/{identifier}", Parent_Experiment_ID=HISTORY_ID,
            Configuration_SHA256=content_hash({"Study_Config": registry["Config_SHA256"], "Candidate": row}), Candidate=row)
        imported("historical-candidate/"+identifier, specification,
                 {"Candidate_ID": identifier, "Observations": len(trials),
                  "Ledger_Net_Excess_Sharpe": float(annualized_sharpe(trials[identifier])),
                  "Artifacts": [_artifact(project, "runs/layer3_candidate_family/trial_returns.parquet")]})
    for layer, purpose in [(2, "accounting_replay"), (3, "statistical_inference"), (4, "attribution")]:
        path = f"runs/layer{layer}_{purpose}/completion.json"
        receipt = json.loads((project/path).read_text())
        imported(f"historical-layer{layer}", dict(base, Parent_Experiment_ID=HISTORY_ID, Purpose=purpose,
            Configuration_SHA256=content_hash(receipt), Source_Revision="sha256-"+content_hash(receipt["Source_SHA256"])),
            {"Artifacts": [_artifact(project, path)], "Interpretation": "Diagnostic recomputation of inspected history"})
    specifications.append({"Type": "family_imported", "Experiment_ID": HISTORY_ID,
        "Idempotency_Key": "import-historical-family-v1.1", "Payload": {
            "Statistical_Family_ID": HISTORY_ID, "Registered_Statistical_Trials": 182,
            "MaxT_Hypotheses_Including_Reported_Selector": 183,
            "Members": ["historical-candidate/"+value for value in candidates.Candidate_ID],
            "Broader_Search_Count": None,
            "History_Completeness": "Earlier informal model, feature, target and allocation attempts are incompletely reconstructable.",
            "Other_Known_Work": {"Locked_Advanced_Model_Families": 4, "Layer2_Portfolio_Cost_Ledgers": 28,
                                 "Layer4_Control_Portfolios": 12, "Randomized_Reference_Draws": 200},
            "Reference_Draw_Interpretation": "Randomized controls are reference draws within one diagnostic procedure.",
            "Artifacts": [_artifact(project, "model_outputs/28_sharpe_experiment_registry.json"),
                          _artifact(project, "model_outputs/32_sharpe_candidate_registry.csv")]}})
    store.append_many(specifications)
    return store.save_anchor()


def comparison_table(states):
    rows = []
    for identifier, state in states.items():
        spec, outcome = state["Specification"], state["Outcome"] or {}
        rows.append({"Experiment_ID": identifier, "Hypothesis_ID": spec["Hypothesis_ID"], "Stage": spec["Stage"],
            "Purpose": spec["Purpose"], "State": state["State"], "Parent_Experiment_ID": spec.get("Parent_Experiment_ID"),
            "Configuration_SHA256": spec["Configuration_SHA256"], "Source_Revision": spec["Source_Revision"],
            "Dataset_IDs": json.dumps(spec["Dataset_IDs"]), "Seed": spec["Seed"],
            "Net_Excess_Sharpe": outcome.get("Ledger_Net_Excess_Sharpe"), "Failure_Reason": outcome.get("Reason"),
            "Corrections": len(state["Corrections"]), "Correction_Details": json.dumps(state["Corrections"]),
            "Artifacts": json.dumps(outcome.get("Artifacts", []))})
    return pd.DataFrame(rows)


def _verify_historical_artifacts(project, events):
    for event in events:
        payload = event["Payload"]
        references = payload.get("Artifacts", [])+payload.get("Outcome", {}).get("Artifacts", [])
        for artifact in references:
            path = project/artifact["Path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["SHA256"]:
                raise ValueError(f"Registered artifact changed: {artifact['Path']}")


def governance_report(project, frozen, store, output):
    project, output = Path(project).resolve(), Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh directory for a governance report")
    verified = verify_freeze(frozen, store, project=project)
    anchor = store.save_anchor()
    with store.transaction():
        events, _, _ = store._read(anchor)
    events = events[:anchor["Sequence"]]
    from .experiments import _apply_event
    states, identifiers = {}, {}
    for event in events:
        _apply_event(event, states, identifiers)
    _verify_historical_artifacts(project, events)
    for event in events:
        if event["Type"] == "evaluation_recorded" and event["Payload"].get("Protocol_SHA256") == verified["Receipt"]["Body"]["Protocol_SHA256"]:
            verify_evaluation(project, event, verified, store)
    table = comparison_table(states)
    selections = pd.read_csv(project/"model_outputs/34_sharpe_outer_selections.csv")
    candidates = pd.read_csv(project/"model_outputs/32_sharpe_candidate_registry.csv")
    lookup = candidates.set_index("Candidate_ID").Is_Benchmark
    selections["Is_Benchmark"] = selections.Selected_Candidate.map(lookup)
    if selections.Is_Benchmark.isna().any():
        raise ValueError("Outer selections include an undisclosed candidate")
    frequency = selections.groupby(["Selection_Label", "Selected_Candidate", "Is_Benchmark"]).size().rename("Selected_Folds").reset_index()
    frequency["Selection_Fraction"] = frequency.Selected_Folds/frequency.groupby("Selection_Label").Selected_Folds.transform("sum")
    evaluations = [event["Payload"] for event in events if event["Type"] == "evaluation_recorded"
                   and event["Payload"].get("Protocol_SHA256") == verified["Receipt"]["Body"]["Protocol_SHA256"]]
    forward_results = [value for value in evaluations if value["Stage"] == "forward"]
    forward = forward_results[-1]["Gates"] if forward_results else assess_gates({}, {"Observations": 0, "Episodes": 0}, verified["Protocol"])
    forward_observations = forward_results[-1]["Metrics"]["Observations"] if forward_results else 0
    report = {"Schema_Version": "1.0.0", "Created_UTC": utc_now(),
        "Protocol_SHA256": verified["Receipt"]["Body"]["Protocol_SHA256"],
        "Stage_Status": {"retrospective": "Imported completed historical studies", "development": "Inspected history available for development",
                         "transfer": "Registered, awaiting complete market data and evaluation", "forward": "Registered, awaiting eligible contemporaneous decisions"},
        "Forward_Gates": forward, "Forward_Observations": forward_observations,
        "Registered_Evaluations": evaluations,
        "Historical_Statistical_Family_Count": 182, "Historical_MaxT_Hypotheses": 183,
        "Known_Experiment_Records": len(states), "Broader_Project_Search_Count": None,
        "History_Limitation": "Earlier informal attempts are incompletely reconstructable. Imported records preserve their retrospective status.",
        "Transfer_Universe": verified["Protocol"]["Universe"]["Transfer"],
        "First_Scheduled_Decision_UTC": verified["Receipt"]["Body"]["First_Scheduled_Decision_UTC"],
        "Evidence_Interpretation": "The existing negative model-increment result is reportable. Forward evidence is inconclusive until the registered horizon and episode minimum are met."}
    output.mkdir(parents=True)
    table.to_csv(output/"experiment_comparison.csv", index=False, lineterminator="\n")
    frequency.to_csv(output/"historical_selection_frequency.csv", index=False, lineterminator="\n")
    atomic_write(output/"governance.json", canonical_bytes(report)+b"\n")
    atomic_write(output/"experiment_anchor.json", canonical_bytes(anchor)+b"\n")
    completion = {"Schema_Version": "1.0.0", "Completed_UTC": utc_now(), "Freeze_Directory": Path(frozen).resolve().relative_to(project).as_posix(),
                  "Protocol_SHA256": report["Protocol_SHA256"], "Log_Anchor": anchor,
                  "Files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()}}
    atomic_write(output/"completion.json", canonical_bytes(completion)+b"\n")
    return report


def verify_report(project, output, store, archived=False):
    project = Path(project).resolve()
    output = Path(output)
    receipt = json.loads((output/"completion.json").read_text())
    store.verify(receipt["Log_Anchor"])
    saved = Path(receipt["Freeze_Directory"])
    if saved.is_absolute():
        # Older receipts stored absolute paths. Relocate the immutable bundle
        # under the requested project, including materialized baseline archives.
        parts = saved.parts
        saved = Path(*parts[parts.index("experiments"):])
    verified = verify_freeze(project/saved, store, project=None if archived else project)
    if verified["Receipt"]["Body"]["Protocol_SHA256"] != receipt["Protocol_SHA256"]:
        raise ValueError("Governance report protocol mismatch")
    for name, digest in receipt["Files"].items():
        if hashlib.sha256((output/name).read_bytes()).hexdigest() != digest:
            raise ValueError("Governance report artifact changed")
    with store.transaction():
        events, states, _ = store._read(receipt["Log_Anchor"])
    _verify_historical_artifacts(Path(project), events[:receipt["Log_Anchor"]["Sequence"]])
    for event in events[:receipt["Log_Anchor"]["Sequence"]]:
        if event["Type"] == "evaluation_recorded" and event["Payload"].get("Protocol_SHA256") == receipt["Protocol_SHA256"]:
            verify_evaluation(project, event, verified, store)
    # Reconstruct only the prefix represented by the saved report.
    from .experiments import _apply_event
    states, identifiers = {}, {}
    for event in events[:receipt["Log_Anchor"]["Sequence"]]:
        _apply_event(event, states, identifiers)
    expected = comparison_table(states).to_csv(index=False, lineterminator="\n").encode()
    if expected != (output/"experiment_comparison.csv").read_text().encode():
        raise ValueError("Comparison table omits or changes a registered experiment")
    return {"Verified": True, "Experiments": len(states), "Anchor_Sequence": receipt["Log_Anchor"]["Sequence"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=".")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("import-history")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("protocol")
    verify = sub.add_parser("verify")
    verify.add_argument("--frozen")
    verify.add_argument("--report")
    verify.add_argument("--archived", action="store_true")
    report = sub.add_parser("report")
    report.add_argument("--frozen", required=True)
    report.add_argument("--output", required=True)
    record = sub.add_parser("record-decision")
    record.add_argument("--frozen", required=True)
    record.add_argument("--experiment", required=True)
    record.add_argument("decision_json")
    args = parser.parse_args()
    project = Path(args.project).resolve()
    store = ExperimentStore(project/"experiments")
    if args.command == "import-history":
        result = import_history(project, store)
    elif args.command == "freeze":
        result = {"Frozen_Directory": str(freeze_protocol(project, project/args.protocol, store))}
    elif args.command == "report":
        result = governance_report(project, project/args.frozen, store, project/args.output)
    elif args.command == "record-decision":
        result = record_forward_decision(project, project/args.frozen, store, args.experiment,
                                         json.loads((project/args.decision_json).read_text()))
    elif args.report:
        result = verify_report(project, project/args.report, store, archived=args.archived)
    elif args.frozen:
        result = verify_freeze(project/args.frozen, store, project=None if args.archived else project)["Receipt"]
    else:
        result = store.verify()
        result.pop("States")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

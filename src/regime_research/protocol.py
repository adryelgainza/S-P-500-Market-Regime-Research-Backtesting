"""Versioned research commitments with archived source, environment, and log heads."""

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import uuid

import pandas as pd

from .data import load_dataset
from .data.calendar import exchange_schedule
from .experiments import ExperimentStore, ZERO_HASH, atomic_write, canonical_bytes, content_hash, publish_directory, utc_now


REQUIRED = {
    "Hypothesis": {"ID", "Economic_Rationale", "Predicted_Behavior", "Primary_Estimand", "Falsification"},
    "Universe": {"Primary", "Transfer", "Currency", "Calendar", "Inception_Rule", "Inclusion_Rationale"},
    "Data": {"Dataset_IDs", "Adjustment_Policy", "Cash_Policy", "Quality_Exclusions", "Availability", "Refresh_Policy"},
    "Models": {"Features", "Target", "Fit_Window", "Preprocessing", "Calibration", "Seed", "Refit_Schedule", "Recipes"},
    "Allocation": {"Candidates", "Selection", "Bounds", "Rebalance", "Overlays", "Benchmark_Eligible", "Comparators", "One_Way_Cost"},
    "Evaluation": {"Development", "Inner_Outer", "Purging", "Transfer", "Forward", "Decision_Delay_Minutes"},
    "Inference": {"Primary_Comparator", "Family", "Confidence", "Uncertainty", "Multiplicity", "Thresholds", "Sensitivity"},
    "Operations": {"Runtime_Lock", "Source_Revision", "Monitoring", "Missing_Data", "Stopping", "Reporting"},
}


def aware_time(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("Research timestamps require an explicit timezone")
    return result.tz_convert("UTC")


def validate_protocol(protocol):
    if protocol.get("Schema_Version") != "1.0.0":
        raise ValueError("Unsupported validation protocol schema")
    for name in ["Protocol_ID", "Version"]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", protocol.get(name, "")):
            raise ValueError(f"Invalid {name}")
    for section, keys in REQUIRED.items():
        value = protocol.get(section)
        if not isinstance(value, dict) or not keys.issubset(value):
            raise ValueError(f"Missing protocol fields in {section}")
        if any(value[key] is None or value[key] == "" for key in keys):
            raise ValueError(f"Empty commitment in {section}")
    allocation, inference = protocol["Allocation"], protocol["Inference"]
    # The initial governance implementation supports a fully fixed rule.
    if allocation["Selection"] != "fixed_single_candidate" or len(allocation["Candidates"]) != 1:
        raise ValueError("This protocol version requires one fixed candidate")
    if allocation["Benchmark_Eligible"] is not False:
        raise ValueError("Fixed-candidate protocols cannot select a benchmark")
    if allocation["Bounds"] != [0, 1] or not 0 <= allocation["One_Way_Cost"] < 1:
        raise ValueError("Unsupported allocation bounds or costs")
    if len(inference["Family"]) != 1 or inference["Family"] != allocation["Candidates"]:
        raise ValueError("The statistical family must match the fixed candidate set")
    if inference["Primary_Comparator"] not in allocation["Comparators"] or not .5 < inference["Confidence"] < 1:
        raise ValueError("Invalid comparator or confidence")
    thresholds = inference["Thresholds"]
    required = {"Minimum_Delta_Sharpe", "Minimum_Observations", "Minimum_Episodes", "Maximum_Drawdown",
                "Adjusted_P_Max", "DSR_Min", "CI_Lower_Min", "Cost_Delta_Sharpe_Min"}
    if not required.issubset(thresholds) or thresholds["Minimum_Observations"] < 100 or thresholds["Minimum_Episodes"] < 1:
        raise ValueError("Evidence thresholds require an adequate time and episode minimum")
    if thresholds["Minimum_Delta_Sharpe"] <= 0 or not 0 < thresholds["Maximum_Drawdown"] < 1:
        raise ValueError("Invalid economic improvement or risk threshold")
    for key in ["Adjusted_P_Max", "DSR_Min"]:
        if not 0 < thresholds[key] < 1:
            raise ValueError("Probability thresholds must lie strictly between zero and one")
    transfer = protocol["Universe"]["Transfer"]
    symbols = [row["Instrument"] for row in transfer]
    if len(set(symbols)) != len(symbols) or protocol["Universe"]["Primary"] in symbols or not transfer:
        raise ValueError("Register a distinct transfer universe")
    if any(row["Weight"] <= 0 or not isinstance(row["Previously_Inspected"], bool) for row in transfer):
        raise ValueError("Transfer markets need positive weights and inspection flags")
    if abs(sum(row["Weight"] for row in transfer)-1) > 1e-12:
        raise ValueError("Transfer weights must sum to one")
    if protocol["Evaluation"]["Decision_Delay_Minutes"] < 1 or not protocol["Data"]["Dataset_IDs"]:
        raise ValueError("Declare data and close-time availability delay")
    if protocol["Evaluation"]["Forward"]["Minimum_Completed_Intervals"] != thresholds["Minimum_Observations"]:
        raise ValueError("Review horizon must match the minimum evidence horizon")
    return protocol


def source_contents(project):
    """Capture scientific code, including notebook code without rendered outputs."""
    project = Path(project)
    paths = sorted(set(project.glob("*.py")) | set((project/"src").rglob("*.py")))
    result = {path.relative_to(project).as_posix(): path.read_bytes() for path in paths}
    for notebook in sorted(project.glob("*.ipynb")):
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        code = [{"id": cell.get("id"), "source": "".join(cell["source"])}
                for cell in cells if cell["cell_type"] == "code"]
        result[notebook.name+".code.json"] = canonical_bytes(code)
    if (project/"pyproject.toml").exists():
        result["pyproject.toml"] = (project/"pyproject.toml").read_bytes()
    if not result:
        raise ValueError("A freeze requires scientific source files")
    return result


def runtime_snapshot(project=None):
    if project is not None and (Path(project)/"requirements.lock").exists():
        from .environment import verify_environment
        runtime = verify_environment(project)
        runtime.update(Platform_System=platform.system(), Machine=platform.machine())
        return runtime, (Path(project)/"requirements.lock").read_bytes()
    packages = {}
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        if name and name.lower().replace("_", "-") != "regime-research":
            packages[name.lower().replace("_", "-")] = distribution.version
    lock = "".join(f"{name}=={version}\n" for name, version in sorted(packages.items()))
    return {"Python": platform.python_version(), "Platform": platform.platform(), "Packages": packages}, lock.encode()


def _finish_bundle(store, event):
    payload = event["Payload"]
    destination = store.directory/"frozen"/payload["Protocol_ID"]/payload["Version"]
    if destination.exists():
        return destination
    pending = store.directory/".pending-freezes"/event["Event_ID"]
    if not pending.exists():
        raise ValueError("Committed freeze snapshot is absent")
    body = json.loads((pending/"receipt_body.json").read_text(encoding="utf-8"))
    if content_hash(body) != payload["Receipt_Body_SHA256"]:
        raise ValueError("Pending freeze receipt differs from its committed event")
    receipt = {"Body": body, "Body_SHA256": content_hash(body),
               "Anchor": {"Sequence": event["Sequence"], "Event_SHA256": event["Event_SHA256"]}}
    atomic_write(pending/"receipt.json", canonical_bytes(receipt)+b"\n")
    destination.parent.mkdir(parents=True, exist_ok=True)
    publish_directory(pending, destination, root=store.directory)
    return destination


def freeze_protocol(project, protocol_path, store=None):
    """Commit a real-time freeze. A retry recovers a committed staged bundle."""
    project = Path(project)
    store = store or ExperimentStore(project/"experiments")
    protocol = validate_protocol(json.loads(Path(protocol_path).read_text(encoding="utf-8")))
    digest = content_hash(protocol)
    identifier, version = protocol["Protocol_ID"], protocol["Version"]
    with store.transaction():
        events, _, _ = store._read()
        matches = [event for event in events if event["Type"] == "protocol_frozen"
                   and event["Payload"]["Protocol_ID"] == identifier and event["Payload"]["Version"] == version]
        if matches:
            if matches[0]["Payload"]["Protocol_SHA256"] != digest:
                raise ValueError("A frozen version cannot be edited. Assign a new version")
            destination = _finish_bundle(store, matches[0])
        else:
            event_id = uuid.uuid4().hex
            pending = store.directory/".pending-freezes"/event_id
            pending.mkdir(parents=True)
            contents = {"source/"+name: data for name, data in source_contents(project).items()}
            source_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
            runtime, lock = runtime_snapshot(project)
            contents["requirements.lock"] = lock
            contents["runtime.json"] = canonical_bytes(runtime)
            contents["protocol.json"] = canonical_bytes(protocol)
            for dataset_id in protocol["Data"]["Dataset_IDs"]:
                load_dataset(dataset_id, root=project/"datasets")
                contents[f"data/{dataset_id}.json"] = (project/"datasets"/dataset_id/"manifest.json").read_bytes()
            files = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
            for name, data in contents.items():
                atomic_write(pending/name, data)
            frozen_at = aware_time(utc_now())
            schedule = exchange_schedule(frozen_at.date(), (frozen_at+pd.Timedelta(days=30)).date(),
                                         protocol["Universe"]["Calendar"])
            decisions = schedule.Close_Time+pd.Timedelta(minutes=protocol["Evaluation"]["Decision_Delay_Minutes"])
            earliest = decisions[decisions > frozen_at].iloc[0]
            body = {"Schema_Version": "1.0.0", "Protocol_ID": identifier, "Version": version,
                    "Protocol_SHA256": digest, "Frozen_UTC": frozen_at.isoformat(),
                    "First_Scheduled_Decision_UTC": earliest.isoformat(),
                    "Source_Revision": "sha256-"+content_hash(source_hashes),
                    "Source_SHA256": source_hashes, "Dependency_Lock_SHA256": files["requirements.lock"],
                    "Dataset_IDs": protocol["Data"]["Dataset_IDs"], "Files": files,
                    "Previous_Log_Head": {"Sequence": len(events), "Event_SHA256": events[-1]["Event_SHA256"] if events else ZERO_HASH},
                    "Freeze_Event_ID": event_id}
            atomic_write(pending/"receipt_body.json", canonical_bytes(body)+b"\n")
            event = store._append_many([{"Type": "protocol_frozen", "Experiment_ID": None, "Event_ID": event_id,
                "Recorded_UTC": frozen_at.isoformat(), "Idempotency_Key": f"freeze-{identifier}-{version}",
                "Payload": {"Protocol_ID": identifier, "Version": version, "Protocol_SHA256": digest,
                            "Receipt_Body_SHA256": content_hash(body)}}])[0]
            destination = _finish_bundle(store, event)
    verify_freeze(destination, store, project=project)
    return destination


def verify_freeze(directory, store, project=None):
    directory = Path(directory)
    receipt = json.loads((directory/"receipt.json").read_text(encoding="utf-8"))
    body = receipt["Body"]
    if content_hash(body) != receipt["Body_SHA256"]:
        raise ValueError("Freeze receipt body was modified")
    with store.transaction():
        events, _, _ = store._read(receipt["Anchor"])
    event = events[receipt["Anchor"]["Sequence"]-1]
    if (event["Type"] != "protocol_frozen" or event["Event_ID"] != body["Freeze_Event_ID"]
            or event["Payload"]["Receipt_Body_SHA256"] != receipt["Body_SHA256"]):
        raise ValueError("Freeze receipt is absent from the anchored experiment history")
    if (event["Recorded_UTC"] != body["Frozen_UTC"]
            or event["Previous_SHA256"] != body["Previous_Log_Head"]["Event_SHA256"]
            or event["Sequence"] != body["Previous_Log_Head"]["Sequence"]+1
            or content_hash(json.loads((directory/"receipt_body.json").read_text(encoding="utf-8"))) != receipt["Body_SHA256"]):
        raise ValueError("Freeze timestamp, parent head or staged receipt mismatch")
    for name, digest in body["Files"].items():
        path = directory/name
        if directory.resolve() not in path.resolve().parents or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen artifact changed: {name}")
    protocol = validate_protocol(json.loads((directory/"protocol.json").read_text(encoding="utf-8")))
    if content_hash(protocol) != body["Protocol_SHA256"]:
        raise ValueError("Frozen protocol hash mismatch")
    if project is not None:
        live = {"source/"+name: hashlib.sha256(data).hexdigest() for name, data in source_contents(project).items()}
        if live != body["Source_SHA256"]:
            raise ValueError("Scientific source changed after freezing. Create a new protocol version")
        runtime, lock = runtime_snapshot(project)
        if hashlib.sha256(lock).hexdigest() != body["Dependency_Lock_SHA256"]:
            raise ValueError("Runtime dependencies differ from the freeze lock")
        saved_runtime = json.loads((directory/"runtime.json").read_text(encoding="utf-8"))
        platform_matches = runtime["Platform"] == saved_runtime["Platform"]
        if "Platform_System" in saved_runtime:
            platform_matches = all(runtime[key] == saved_runtime[key] for key in ["Platform_System", "Machine"])
        if runtime["Python"] != saved_runtime["Python"] or not platform_matches:
            raise ValueError("Python or platform differs from the frozen runtime")
    return {"Protocol": protocol, "Receipt": receipt}

"""Supported offline entry points and strict fresh-kernel completion receipts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

import nbformat
from jupyter_client import KernelManager

from .archive import file_hash, materialize_baseline, verify_archive
from .config import RunConfig
from .environment import verify_environment
from .experiments import ExperimentStore, ZERO_HASH, atomic_write, canonical_bytes, content_hash, utc_now
from .protocol import source_contents, validate_protocol, verify_freeze
from .research import verify_study


def scientific_sources(project):
    return {name: hashlib.sha256(value).hexdigest() for name, value in source_contents(project).items()}


def git_revision(project):
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project, capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=project,
                            capture_output=True, text=True, check=True).stdout
    return {"Commit": revision, "Working_Tree_Clean": not bool(status.strip())}


def execution_plan(notebook):
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    identifiers = [cell.get("id") for cell in cells]
    if not cells or any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Notebook code cells need unique persistent IDs")
    if any(not cell.source.strip() for cell in cells):
        raise ValueError("Remove empty executable cells before a supported run")
    return [{"Cell_ID": cell.id, "Source": cell.source, "Source_SHA256": hashlib.sha256(cell.source.encode()).hexdigest()}
            for cell in cells]


def verify_kernel_audit(plan, audit):
    previous = ZERO_HASH
    if len(audit) != 2*len(plan):
        raise ValueError("Kernel audit contains missing or unexpected executions")
    for index, event in enumerate(audit):
        body = {key: value for key, value in event.items() if key != "Event_SHA256"}
        if event["Previous_SHA256"] != previous or content_hash(body) != event["Event_SHA256"]:
            raise ValueError("Kernel audit hash chain mismatch")
        previous = event["Event_SHA256"]
        cell_number = index//2
        expected_type = "cell_started" if index % 2 == 0 else "cell_completed"
        planned = plan[cell_number]
        if (event["Type"] != expected_type or event["Index"] != cell_number
                or event["Cell_ID"] != planned["Cell_ID"] or event["Source_SHA256"] != planned["Source_SHA256"]):
            raise ValueError("Kernel execution differs from its exact saved sequence")
        if index % 2 and event["Execution_Count"] != cell_number+1:
            raise ValueError("Kernel history has an unexpected execution count")
    return previous


def execute_cells(project, notebook_path, output, *, timeout=1800, source_check=None, environment=None):
    """Launch one private kernel and execute each saved cell exactly once in order."""
    project, output = Path(project).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(notebook_path, as_version=4)
    plan = execution_plan(notebook)
    plan_path, audit_path = output/"cell_plan.json", output/"kernel_audit.jsonl"
    atomic_write(plan_path, canonical_bytes(plan)+b"\n")
    kernel_directory = output/"kernel"
    kernel_directory.mkdir()
    env = dict(os.environ, REGIME_CELL_PLAN=str(plan_path), REGIME_KERNEL_AUDIT=str(audit_path),
        REGIME_PROJECT=str(project), REGIME_RUN_OUTPUT=str(output), REGIME_RUN_CONFIG=str(output/"configuration.json"),
        IPYTHONDIR=str(kernel_directory/"ipython"), JUPYTER_RUNTIME_DIR=str(kernel_directory),
        MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="0")
    env.update(environment or {})
    manager = KernelManager(kernel_name="python3", connection_file=str(kernel_directory/"connection.json"))
    manager.kernel_spec.argv = [sys.executable, "-m", "regime_research.kernel", "-f", "{connection_file}"]
    client = None
    records = []
    stdout = (output/"kernel_stdout.log").open("w")
    stderr = (output/"kernel_stderr.log").open("w")
    try:
        manager.start_kernel(cwd=str(project), env=env, stdout=stdout, stderr=stderr)
        client = manager.blocking_client()
        client.start_channels()
        client.wait_for_ready(timeout=60)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        for index, (planned, cell) in enumerate(zip(plan, code_cells)):
            if source_check is not None:
                source_check()
            started = utc_now()
            message_id = client.execute(planned["Source"], silent=False, store_history=True,
                                        allow_stdin=False, stop_on_error=True)
            outputs, observed_input = [], []
            deadline = time.monotonic()+timeout
            while True:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Cell timed out: {planned['Cell_ID']}")
                try:
                    message = client.get_iopub_msg(timeout=min(1, remaining))
                except queue.Empty:
                    if not manager.is_alive():
                        raise RuntimeError("Research kernel terminated before completion")
                    continue
                kind, parent, content = message["msg_type"], message.get("parent_header", {}), message["content"]
                if kind == "execute_input" and parent.get("msg_id") != message_id:
                    raise ValueError("Kernel received an unexpected external execute request")
                if parent.get("msg_id") != message_id:
                    continue
                if kind == "execute_input":
                    observed_input.append(content)
                elif kind in {"stream", "display_data", "execute_result", "error"}:
                    fields = {key: value for key, value in content.items() if key in
                              {"name", "text", "data", "metadata", "execution_count", "ename", "evalue", "traceback"}}
                    outputs.append(nbformat.v4.new_output(kind, **fields))
                elif kind == "clear_output":
                    outputs = []
                if kind == "status" and content["execution_state"] == "idle":
                    break
            while True:
                reply = client.get_shell_msg(timeout=max(1, deadline-time.monotonic()))
                if reply["msg_type"] == "execute_reply":
                    if reply.get("parent_header", {}).get("msg_id") != message_id:
                        raise ValueError("Kernel replied to an unexpected execute request")
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for the cell execution reply")
            if reply["content"]["status"] != "ok":
                cell.outputs = outputs
                raise RuntimeError(f"Notebook cell failed: {planned['Cell_ID']} {reply['content']}")
            if (len(observed_input) != 1 or observed_input[0]["code"] != planned["Source"]
                    or observed_input[0]["execution_count"] != index+1):
                raise ValueError("Observed kernel input differs from the saved cell")
            cell.outputs, cell.execution_count = outputs, index+1
            records.append({"Cell_ID": planned["Cell_ID"], "Source_SHA256": planned["Source_SHA256"],
                "Execution_Count": index+1, "Request_ID": message_id, "Kernel_Session": reply["header"]["session"],
                "Started_UTC": started, "Completed_UTC": utc_now()})
            atomic_write(output/"execution_progress.json", canonical_bytes({"Completed_Cells": records})+b"\n")
            print(f"Completed notebook cell {index+1}/{len(plan)} {planned['Cell_ID']}", flush=True)
    finally:
        if client is not None:
            client.stop_channels()
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        stdout.close()
        stderr.close()
        nbformat.write(notebook, output/"executed_notebook.ipynb")
    if source_check is not None:
        source_check()
    audit_raw = audit_path.read_bytes()
    if not audit_raw.endswith(b"\n"):
        raise ValueError("Kernel audit has an interrupted record")
    audit = [json.loads(line) for line in audit_raw.splitlines()]
    head = verify_kernel_audit(plan, audit)
    if len({row["Kernel_Session"] for row in records}) != 1:
        raise ValueError("Notebook execution used multiple kernel sessions")
    receipt = {"Plan_SHA256": content_hash(plan), "Cells": records, "Kernel_Audit_Head": head,
               "Kernel_Audit_SHA256": file_hash(audit_path), "Strict_Sequence": True}
    atomic_write(output/"execution_receipt.json", canonical_bytes(receipt)+b"\n")
    return receipt


def run_notebook(project, output, config, notebook_path=None):
    project, output = Path(project).resolve(), Path(output).resolve()
    if not output.is_relative_to(project):
        raise ValueError("Research run directories must lie inside the project")
    if output.exists():
        raise FileExistsError("A notebook run requires a fresh output directory")
    output.mkdir(parents=True)
    notebook_path = Path(notebook_path or project/"sp500_market_regime_classification.ipynb").resolve()
    if notebook_path != project/"sp500_market_regime_classification.ipynb":
        raise ValueError("The supported release runner executes the canonical project notebook")
    environment = verify_environment(project)
    protocol = validate_protocol(json.loads((project/config.protocol).read_text(encoding="utf-8")))
    source = scientific_sources(project)
    revision = git_revision(project)
    lock_hash = file_hash(project/"requirements.lock")
    snapshot = {"Sources": source, "Protocol_SHA256": content_hash(protocol), "Lock_SHA256": lock_hash}
    atomic_write(output/"configuration.json", canonical_bytes(config.to_dict())+b"\n")
    store = ExperimentStore(project/"experiments")
    spec = {"Hypothesis_ID": protocol["Hypothesis"]["ID"], "Configuration_SHA256": content_hash(config.to_dict()),
            "Protocol_SHA256": content_hash(protocol), "Source_Revision": "sha256-"+content_hash(source),
            "Dataset_IDs": protocol["Data"]["Dataset_IDs"], "Stage": "retrospective" if config.mode == "replay" else "development",
            "Seed": config.seed, "Purpose": "strict fresh-kernel "+config.mode}
    def unchanged():
        actual = {"Sources": scientific_sources(project),
                  "Protocol_SHA256": content_hash(json.loads((project/config.protocol).read_text(encoding="utf-8"))),
                  "Lock_SHA256": file_hash(project/"requirements.lock")}
        if actual != snapshot:
            raise ValueError("Scientific source, notebook, protocol or lock changed during execution")
    try:
        with store.attempt(spec) as (identifier, outcome):
            execute_cells(project, notebook_path, output, timeout=config.cell_timeout_seconds, source_check=unchanged)
            verification = verify_study(output/"study", config.numerical_rtol, config.numerical_atol, project=project)
            unchanged()
            freeze = project/"experiments/frozen"/protocol["Protocol_ID"]/protocol["Version"]
            freeze_receipt = verify_freeze(freeze, store, project=project)["Receipt"] if freeze.exists() else None
            files = {path.relative_to(output).as_posix(): file_hash(path) for path in output.rglob("*")
                     if path.is_file() and "kernel" not in path.relative_to(output).parts}
            anchor = store.save_anchor()
            completion = {"Schema_Version": "2.0.0", "Run_ID": identifier, "Completed_UTC": utc_now(),
                "Stage": spec["Stage"], "Git": revision, "Source_Revision": spec["Source_Revision"],
                "Source_SHA256": source, "Configuration": config.to_dict(), "Protocol": protocol,
                "Protocol_SHA256": content_hash(protocol), "Freeze_Receipt": freeze_receipt,
                "Environment": environment, "Package_Version": environment["Package_Version"],
                "Dataset_IDs": spec["Dataset_IDs"], "Log_Anchor": anchor,
                "Baseline_Archive_SHA256": verify_archive(project)["Archive_SHA256"],
                "Study_Verification": verification, "Files": files}
            atomic_write(output/"pending_completion.json", canonical_bytes(completion)+b"\n")
            verify_run(project, output, manifest_name="pending_completion.json")
            atomic_write(output/"completion.json", canonical_bytes(completion)+b"\n")
            (output/"pending_completion.json").unlink()
            outcome["Artifacts"] = [{"Path": (output/"completion.json").relative_to(project).as_posix(),
                                     "SHA256": file_hash(output/"completion.json")}]
            outcome["Result"] = verification
    except BaseException as error:
        (output/"completion.json").unlink(missing_ok=True)
        atomic_write(output/"failure.json", canonical_bytes({"Failed_UTC": utc_now(),
                     "Reason": f"{type(error).__name__}: {error}"})+b"\n")
        raise
    return completion


def verify_run(project, output, manifest_name="completion.json"):
    project, output = Path(project).resolve(), Path(output).resolve()
    manifest = json.loads((output/manifest_name).read_text(encoding="utf-8"))
    if manifest["Schema_Version"] != "2.0.0" or manifest["Source_SHA256"] != scientific_sources(project):
        raise ValueError("Completed run uses different scientific source")
    if manifest["Source_Revision"] != "sha256-"+content_hash(manifest["Source_SHA256"]):
        raise ValueError("Completed run source revision does not reconcile")
    if content_hash(manifest["Protocol"]) != manifest["Protocol_SHA256"]:
        raise ValueError("Run protocol checksum mismatch")
    if content_hash(json.loads((project/manifest["Configuration"]["protocol"]).read_text(encoding="utf-8"))) != manifest["Protocol_SHA256"]:
        raise ValueError("Current protocol differs from the completed run")
    if verify_archive(project)["Archive_SHA256"] != manifest["Baseline_Archive_SHA256"]:
        raise ValueError("Baseline archive differs from the completed run")
    current = verify_environment(project)
    if current["Packages"] != manifest["Environment"]["Packages"] or current["Lock_SHA256"] != manifest["Environment"]["Lock_SHA256"]:
        raise ValueError("Run dependency lock differs from the supported environment")
    store = ExperimentStore(project/"experiments")
    history = store.verify(manifest["Log_Anchor"])
    if manifest_name == "completion.json":
        state = history["States"].get(manifest["Run_ID"])
        if state is None or state["State"] != "completed":
            raise ValueError("Run has no completed experiment transition")
        specification = state["Specification"]
        if (specification["Source_Revision"] != manifest["Source_Revision"]
                or specification["Configuration_SHA256"] != content_hash(manifest["Configuration"])
                or specification["Protocol_SHA256"] != manifest["Protocol_SHA256"]
                or specification["Stage"] != manifest["Stage"]):
            raise ValueError("Run registration differs from its completed manifest")
        if not any(artifact["Path"] == (output/manifest_name).relative_to(project).as_posix()
                   and artifact["SHA256"] == file_hash(output/manifest_name)
                   for artifact in (state["Outcome"] or {}).get("Artifacts", [])):
            raise ValueError("Completed manifest differs from its recorded artifact hash")
    if manifest["Freeze_Receipt"] is not None:
        protocol = manifest["Protocol"]
        frozen = project/"experiments/frozen"/protocol["Protocol_ID"]/protocol["Version"]
        verified = verify_freeze(frozen, store, project=project)
        if verified["Receipt"] != manifest["Freeze_Receipt"]:
            raise ValueError("Run freeze receipt differs from the anchored protocol")
    for name, digest in manifest["Files"].items():
        path = (output/name).resolve()
        if not path.is_relative_to(output) or file_hash(path) != digest:
            raise ValueError(f"Completed run artifact changed at {name}")
    notebook = nbformat.read(output/"executed_notebook.ipynb", as_version=4)
    plan = execution_plan(notebook)
    saved = json.loads((output/"execution_receipt.json").read_text(encoding="utf-8"))
    if content_hash(plan) != saved["Plan_SHA256"]:
        raise ValueError("Executed notebook cell plan changed")
    original = nbformat.read(project/"sp500_market_regime_classification.ipynb", as_version=4)
    if plan != execution_plan(original):
        raise ValueError("Executed notebook differs from the current canonical notebook")
    audit = [json.loads(line) for line in (output/"kernel_audit.jsonl").read_bytes().splitlines()]
    if verify_kernel_audit(plan, audit) != saved["Kernel_Audit_Head"]:
        raise ValueError("Kernel audit head differs from its receipt")
    if [row["Cell_ID"] for row in saved["Cells"]] != [cell["Cell_ID"] for cell in plan]:
        raise ValueError("Executed cell record is incomplete")
    if (len({row["Kernel_Session"] for row in saved["Cells"]}) != 1
            or len({row["Request_ID"] for row in saved["Cells"]}) != len(plan)
            or [row["Execution_Count"] for row in saved["Cells"]] != list(range(1, len(plan)+1))):
        raise ValueError("Executed request identities or counts are inconsistent")
    verify_study(output/"study", manifest["Configuration"]["numerical_rtol"], manifest["Configuration"]["numerical_atol"], project=project)
    return {"Verified": True, "Run_ID": manifest["Run_ID"], "Stage": manifest["Stage"], "Cells": len(plan)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=".")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--mode", choices=["replay", "fit"], default="replay")
    run.add_argument("--config")
    run.add_argument("--output", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("output")
    sub.add_parser("verify-archive")
    sub.add_parser("restore-baseline")
    args = parser.parse_args()
    project = Path(args.project).resolve()
    if args.command == "run":
        config = RunConfig.from_json(project/args.config) if args.config else RunConfig(mode=args.mode)
        result = run_notebook(project, project/args.output, config)
        result = {"Run_ID": result["Run_ID"], "Stage": result["Stage"], "Output": args.output}
    elif args.command == "verify":
        result = verify_run(project, project/args.output)
    elif args.command == "restore-baseline":
        result = {"Baseline": str(materialize_baseline(project))}
    else:
        manifest = verify_archive(project)
        result = {"Verified": True, "Archive_SHA256": manifest["Archive_SHA256"], "Files": len(manifest["Files"])}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

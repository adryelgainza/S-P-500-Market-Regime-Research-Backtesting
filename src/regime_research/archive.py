"""Materialize and independently replay the immutable pre-extraction baseline."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import zipfile

from .experiments import publish_directory


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(project):
    project = Path(project).resolve()
    manifest = json.loads((project/"archives/layer5_baseline/manifest.json").read_text(encoding="utf-8"))
    path = project/manifest["Archive"]
    if not path.is_relative_to(project) or file_hash(path) != manifest["Archive_SHA256"]:
        raise ValueError("Baseline archive checksum mismatch")
    with zipfile.ZipFile(path) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())) or set(archive.namelist()) != set(manifest["Files"]):
            raise ValueError("Baseline archive member set differs from its manifest")
        for name, digest in manifest["Files"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts or hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"Invalid baseline member {name}")
    return manifest


def materialize_baseline(project, cache=None):
    project = Path(project).resolve()
    manifest = verify_archive(project)
    cache = Path(cache or project/"checkpoints/baseline").resolve()
    destination = cache/manifest["Archive_SHA256"]
    if not destination.exists():
        pending = cache/(".pending-"+uuid.uuid4().hex)
        pending.mkdir(parents=True)
        with zipfile.ZipFile(project/manifest["Archive"]) as archive:
            archive.extractall(pending)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Unique staging and a completed destination make concurrent readers safe.
        try:
            publish_directory(pending, destination, root=cache)
        except FileExistsError:
            pass
    for name, digest in manifest["Files"].items():
        if file_hash(destination/name) != digest:
            raise ValueError(f"Materialized baseline changed at {name}")
    return destination


def replay_baseline(project, output):
    """Run historical verifiers with their archived source and declared numerical pins."""
    baseline = materialize_baseline(project)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    code = """
import json
from pathlib import Path
import verify_sharpe_research
import sys
from regime_research.accounting_replay import verify_accounting_replay
from regime_research.statistical_research import verify_statistical_research
from regime_research.attribution_research import verify_attribution_research
from regime_research.experiments import ExperimentStore
from regime_research.protocol import verify_freeze
p = Path.cwd()
result = {}
result['Accounting'] = verify_accounting_replay(p, p/'runs/layer2_accounting_replay')
result['Inference'] = verify_statistical_research(p, p/'runs/layer3_statistical_inference')
result['Attribution'] = verify_attribution_research(p, p/'runs/layer4_attribution')
store = ExperimentStore(p/'experiments')
receipt = json.loads((p/'runs/layer5_validation_governance/completion.json').read_text())
store.verify(receipt['Log_Anchor'])
verify_freeze(p/'experiments/frozen/downside-overlay/1.0.0', store)
result['Governance'] = {'Anchored_Events': receipt['Log_Anchor']['Sequence']}
from regime_research.experiments import atomic_write, canonical_bytes
atomic_write(Path(sys.argv[1]), canonical_bytes(result)+b'\\n')
"""
    environment = dict(os.environ, PYTHONPATH=str(baseline/"src"), PYTHONDONTWRITEBYTECODE="1")
    with (output/"baseline_verification.log").open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-c", code, str((output/"baseline_verification.json").resolve())], cwd=baseline, env=environment,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((output/"baseline_verification.json").read_text(encoding="utf-8"))
    return baseline, result

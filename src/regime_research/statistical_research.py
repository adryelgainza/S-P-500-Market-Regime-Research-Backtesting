"""Run and verify Layer 3 inference over the frozen, ledger-accounted trial family."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import tempfile

import numpy as np
import pandas as pd

from .statistics import InferenceConfig, deflated_sharpe_from_moments, infer_performance
from .trial_returns import load_trial_family


TABLES = {"dsr_inputs": "dsr_inputs", "trial_dependence": "trial_dependence",
          "trial_diagnostics": "trial_diagnostics", "bootstrap_summary": "bootstrap_summary",
          "hac_summary": "hac_summary", "max_t": "multiple_testing"}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_hashes(base, entries):
    base = Path(base).resolve()
    for name, expected in entries.items():
        path = (base/name).resolve()
        if not path.is_relative_to(base) or _digest(path) != expected:
            raise ValueError(f"Statistical artifact or source checksum mismatch for {name}")


def _verify_calibration(project, calibration):
    manifest = json.loads((calibration/"completion.json").read_text(encoding="utf-8"))
    _verify_hashes(calibration, manifest["Files"])
    _verify_hashes(project/"src/regime_research", manifest["Source_SHA256"])
    if manifest["Protocol_SHA256"] != _digest(project/"protocols/layer3_null_calibration.json"):
        raise ValueError("Calibration protocol differs from the declared protocol")
    if not manifest["All_Passed"]:
        raise ValueError("Calibration failed its declared acceptance gates")
    return manifest


def _paper_example(project):
    fixture = json.loads((project/"tests/fixtures/dsr_published_example.json").read_text(encoding="utf-8"))
    rows = []
    for case in fixture["Cases"]:
        row = deflated_sharpe_from_moments(fixture["Annual_Sharpe"]/np.sqrt(fixture["Annualization"]),
            fixture["Observations"], case["Skewness"], case["Nonexcess_Kurtosis"],
            fixture["Annual_Trial_Sharpe_Variance"]/fixture["Annualization"], case["Independent_Trials"])
        if round(row["DSR"], 4) != case["Expected_DSR_Rounded"]:
            raise ArithmeticError("Published DSR example failed")
        rows.append(dict(**row, Expected_DSR_Rounded=case["Expected_DSR_Rounded"]))
    return dict(Source_URL=fixture["Source_URL"], Source_Pages=fixture["Source_Pages"],
                Annualization=fixture["Annualization"], Passed=True, Cases=rows)


def run_statistical_research(project_dir, family_dir, calibration_dir, output):
    project, output = Path(project_dir).resolve(), Path(output).resolve()
    family, calibration = (project/family_dir).resolve(), (project/calibration_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Statistical output already exists at {output}")
    _verify_calibration(project, calibration)
    portfolio, benchmarks, trials, family_manifest = load_trial_family(project, family)
    config = InferenceConfig(registered_trials=family_manifest["Registered_Trials"])
    source_paths = [Path(__file__), Path(__file__).with_name("statistics.py"),
        Path(__file__).with_name("statistics_calibration.py"), Path(__file__).with_name("trial_returns.py"),
        project/"tests/fixtures/dsr_published_example.json", project/"protocols/layer3_null_calibration.json",
        family/"completion.json", calibration/"completion.json"]
    source_paths.extend(family/name for name in family_manifest["Files"])
    source_hashes = {str(path.relative_to(project).as_posix()): _digest(path) for path in source_paths}
    result = infer_performance(portfolio, benchmarks, trials, config)
    summary = dict(result.inference)
    summary["Reported_Portfolio_MaxT"] = json.loads(result.max_t.loc[
        result.max_t.Candidate_ID.eq("Reported_Portfolio")].to_json(orient="records"))[0]
    summary["Calibration_Passed"] = True
    summary["Calibration_Observed_Rates"] = json.loads(pd.read_csv(calibration/"null_calibration.csv").to_json(orient="records"))
    summary["Evidence_Type"] = "retrospective_nested_pseudo_out_of_sample_with_conditional_return_stream_inference"
    example = _paper_example(project)
    if any(_digest(path) != source_hashes[str(path.relative_to(project).as_posix())] for path in source_paths):
        raise RuntimeError("Statistical inputs changed during inference")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pending-inference-", dir=output.parent) as temporary:
        pending = Path(temporary)/output.name
        pending.mkdir()
        for attribute, filename in TABLES.items():
            getattr(result, attribute).to_csv(pending/f"{filename}.csv", index=False)
        (pending/"inference.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        (pending/"published_example_check.json").write_text(json.dumps(example, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        (pending/"null_calibration.csv").write_bytes((calibration/"null_calibration.csv").read_bytes())
        manifest = dict(Schema_Version="1.0.0", Completed_UTC=datetime.now(timezone.utc).isoformat(),
            Family_Directory=str(family.relative_to(project).as_posix()),
            Calibration_Directory=str(calibration.relative_to(project).as_posix()),
            Inference_Config=asdict(config), Source_SHA256=source_hashes,
            Runtime={"Python": platform.python_version(), **{name: importlib.metadata.version(name)
                     for name in ["numpy", "pandas", "scipy"]}},
            Files={path.name: _digest(path) for path in pending.iterdir()})
        (pending/"completion.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        pending.rename(output)
    return summary


def _compare_saved_table(expected, actual, rtol=1e-10, atol=1e-12):
    # CSV empty fields deserialize as NaN, while in-memory diagnostics use None.
    expected = expected.astype(object).where(expected.notna(), None)
    actual = actual.astype(object).where(actual.notna(), None)
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_exact=False,
                                   rtol=rtol, atol=atol)


def verify_statistical_research(project_dir, output):
    project, output = Path(project_dir).resolve(), Path(output).resolve()
    manifest = json.loads((output/"completion.json").read_text(encoding="utf-8"))
    _verify_hashes(project, manifest["Source_SHA256"])
    _verify_hashes(output, manifest["Files"])
    _verify_calibration(project, project/manifest["Calibration_Directory"])
    portfolio, benchmarks, trials, _ = load_trial_family(project, project/manifest["Family_Directory"])
    config = InferenceConfig(**manifest["Inference_Config"])
    reproduced = infer_performance(portfolio, benchmarks, trials, config)
    for attribute, filename in TABLES.items():
        expected = pd.read_csv(output/f"{filename}.csv", float_precision="round_trip")
        actual = getattr(reproduced, attribute)
        _compare_saved_table(expected, actual)
    _paper_example(project)
    return dict(Verified=True, Observations=len(portfolio), Registered_Trials=len(trials.columns),
                Sharpe=reproduced.inference["Sharpe"], DSR=json.loads(reproduced.dsr_inputs.to_json(orient="records")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--family", default="runs/layer3_candidate_family")
    parser.add_argument("--calibration", default="runs/layer3_null_calibration")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output")
    action.add_argument("--verify")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_statistical_research(args.project_dir, args.verify), indent=2))
    else:
        summary = run_statistical_research(args.project_dir, args.family, args.calibration, args.output)
        print(json.dumps({key: summary[key] for key in ["Observations", "Sharpe", "Trial_Family", "Reported_Portfolio_MaxT"]}, indent=2))


if __name__ == "__main__":
    main()

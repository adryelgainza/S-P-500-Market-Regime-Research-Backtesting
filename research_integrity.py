"""Historical timing and metric helpers retained for archived compatibility.

Supported release execution uses regime_research.runner and its strict receipts.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def json_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_digest(frame):
    """Hash values, row order, column names, and dtypes of every supplied input."""
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns)).encode("utf-8"))
    digest.update(json.dumps(frame.dtypes.astype(str).tolist()).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def notebook_code_cells(notebook_path):
    notebook = json.loads(Path(notebook_path).read_text(encoding="utf-8"))
    return [
        {"id": cell["id"], "source": "".join(cell["source"]).rstrip()}
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def source_provenance(project_dir):
    project_dir = Path(project_dir)
    return {
        "Data_SHA256": file_digest(project_dir / "sp500_ohlcv.csv"),
        "Notebook_Code_SHA256": json_digest(
            notebook_code_cells(project_dir / "sp500_market_regime_classification.ipynb")
        ),
        "Source_SHA256": {
            name: file_digest(project_dir / name)
            for name in ("sharpe_improvement_research.py", "research_integrity.py")
        },
        "Runtime": {
            "Python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scipy", "scikit-learn", "ipython")
            },
        },
    }


def executed_notebook_digest(notebook_path, input_history, through_cell_id):
    """Legacy ordered-history check for archived notebook compatibility."""
    expected = notebook_code_cells(notebook_path)
    stop = next(i for i, cell in enumerate(expected) if cell["id"] == through_cell_id)
    expected = expected[: stop + 1]
    history = iter(source.rstrip() for source in input_history)
    for cell in expected:
        if not any(source == cell["source"] for source in history):
            raise RuntimeError(
                f"Current source for cell {cell['id']} has not executed in order. "
                "Restart the kernel and run all cells."
            )
    return json_digest(expected)


def equity_drawdown(equity, initial_equity=1.0):
    """Include the initial capital as a running-peak candidate."""
    equity = pd.Series(equity, copy=False)
    return equity / equity.cummax().clip(lower=initial_equity) - 1.0


def downside_deviation(returns, target=0.0):
    """Root mean square shortfall over all observations, including zero shortfalls."""
    shortfall = np.minimum(np.asarray(returns, dtype=float) - target, 0.0)
    return float(np.sqrt(np.mean(shortfall**2))) if len(shortfall) else 0.0


def forward_adverse_return(opens, lows, horizon):
    """Worst marked price from entry open through exit open, excluding exit-day lows."""
    if horizon < 1:
        raise ValueError("The holding horizon must be positive")
    path = pd.concat(
        [*[lows.shift(-day) for day in range(1, horizon + 1)],
         opens.shift(-(horizon + 1))],
        axis="columns",
    )
    return path.min(axis="columns", skipna=False) / opens.shift(-1) - 1.0


def realized_selection_rows(frame, start, end, decision_date):
    """Select signal rows only when their full daily P&L was available to selection."""
    return frame.loc[
        frame["Date"].ge(start)
        & frame["Date"].lt(end)
        & frame["Return_End_Date"].lt(decision_date)
    ]


def load_predictor_cache(score_path, audit_path, manifest_path, signature,
                         recipe_ids, expected_dates):
    """Reject absent, stale, damaged, incomplete, or unpurged predictor caches."""
    paths = [Path(score_path), Path(audit_path), Path(manifest_path)]
    if not all(path.is_file() for path in paths):
        return None
    try:
        manifest = json.loads(paths[2].read_text(encoding="utf-8"))
        if manifest.get("Predictor_Cache_Signature") != signature:
            return None
        if manifest.get("Score_SHA256") != file_digest(paths[0]):
            return None
        if manifest.get("Audit_SHA256") != file_digest(paths[1]):
            return None
        scores = pd.read_csv(paths[0], parse_dates=[
            "Date", "Execution_Date", "Return_End_Date", "Label_End_Date", "Refit_Date"
        ])
        audit = pd.read_csv(paths[1], parse_dates=[
            "Refit_Date", "Prediction_End", "Training_Start", "Training_End",
            "Training_Label_End", "Calibration_Start", "Calibration_Label_End"
        ])
        expected_dates = pd.DatetimeIndex(expected_dates)
        if set(scores["Recipe_ID"]) != set(recipe_ids) or set(audit["Recipe_ID"]) != set(recipe_ids):
            return None
        if len(scores) != manifest["Score_Rows"] or len(audit) != manifest["Audit_Rows"]:
            return None
        if scores.duplicated(["Recipe_ID", "Date"]).any() or not np.isfinite(scores["Risk_Score"]).all():
            return None
        for _, group in scores.groupby("Recipe_ID"):
            if not pd.DatetimeIndex(group.sort_values("Date")["Date"]).equals(expected_dates):
                return None
        if not scores["Refit_Date"].le(scores["Date"]).all():
            return None
        if not audit["Training_Label_End"].lt(audit["Refit_Date"]).all():
            return None
        calibrated = audit.loc[audit["Calibration_Rows"].gt(0)]
        if not calibrated["Training_Label_End"].lt(calibrated["Calibration_Start"]).all():
            return None
        if not calibrated["Calibration_Label_End"].lt(calibrated["Refit_Date"]).all():
            return None
    except (ValueError, KeyError, OSError, TypeError):
        return None
    return scores, audit


def write_run_manifest(project_dir, provenance, executed_digest):
    """Publish a completion receipt only after all numbered outputs have been saved."""
    project_dir = Path(project_dir)
    if source_provenance(project_dir) != provenance:
        raise RuntimeError("Research inputs changed during execution. Rerun all cells.")
    output_dir = project_dir / "model_outputs"
    artifacts = {
        path.name: file_digest(path)
        for path in sorted(output_dir.iterdir())
        if path.name[:2].isdigit()
        and path.suffix in {".csv", ".json", ".png"}
        and path.name != "46_sharpe_run_manifest.json"
    }
    manifest = {
        "Run_ID": uuid.uuid4().hex,
        "Completed_UTC": datetime.now(timezone.utc).isoformat(),
        "Provenance": provenance,
        "Executed_Through_Cell": "sharpe-experiments",
        "Executed_Code_SHA256": executed_digest,
        "Artifact_SHA256": artifacts,
    }
    manifest_path = output_dir / "46_sharpe_run_manifest.json"
    pending_path = manifest_path.with_suffix(".json.pending")
    pending_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pending_path.replace(manifest_path)
    return manifest["Run_ID"]


def verify_run_provenance(project_dir):
    """Validate current inputs, artifact bytes, and the saved completion receipt."""
    project_dir = Path(project_dir)
    output_dir = project_dir / "model_outputs"
    manifest = json.loads(
        (output_dir / "46_sharpe_run_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["Provenance"] != source_provenance(project_dir):
        raise AssertionError("Saved research inputs differ from current source or data")
    cells = notebook_code_cells(project_dir / "sp500_market_regime_classification.ipynb")
    stop = next(i for i, cell in enumerate(cells) if cell["id"] == manifest["Executed_Through_Cell"])
    if json_digest(cells[: stop + 1]) != manifest["Executed_Code_SHA256"]:
        raise AssertionError("Saved notebook code differs from executed analysis")
    for name, expected_hash in manifest["Artifact_SHA256"].items():
        if Path(name).name != name or file_digest(output_dir / name) != expected_hash:
            raise AssertionError(f"Research artifact checksum mismatch for {name}")
    notebook = json.loads(
        (project_dir / "sp500_market_regime_classification.ipynb").read_text(encoding="utf-8")
    )
    sharpe_cell = next(cell for cell in notebook["cells"] if cell.get("id") == "sharpe-experiments")
    output_text = "".join(
        "".join(output.get("text", [])) for output in sharpe_cell.get("outputs", [])
    )
    if f"Research run completed: {manifest['Run_ID']}" not in output_text:
        raise AssertionError("Notebook lacks the completed run's execution receipt")
    return manifest

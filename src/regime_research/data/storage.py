"""Content-addressed, atomic snapshots with offline checksum verification."""

from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import tempfile

import numpy as np
import pandas as pd

from .cash import CashPolicy, build_cash_intervals
from .contracts import (Dataset, DataValidationError, SCHEMA_VERSION, TABLE_COLUMNS,
                        TIMESTAMP_COLUMNS, ValidationPolicy, empty_table, utc_timestamp)
from .reconciliation import reconcile_total_returns
from .validation import validate_tables


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _coerce(frame):
    frame = frame.drop(columns=["Dataset_ID"], errors="ignore").copy().reset_index(drop=True)
    for column in set(frame.columns) & TIMESTAMP_COLUMNS:
        for value in frame[column]:
            utc_timestamp(value)
        frame[column] = pd.to_datetime(frame[column], utc=True).astype("datetime64[ns, UTC]")
    return frame


def _csv_bytes(frame):
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")


def _read_table(path):
    frame = pd.read_csv(path, keep_default_na=False, float_precision="round_trip",
                        dtype={c: str for c in ["Session", "Instrument_ID", "Currency", "Source_ID",
                                               "Ex_Date", "Pay_Date", "Rate_Date", "Action_ID",
                                               "Observation_ID", "Observation_IDs"]})
    for column in set(frame.columns) & TIMESTAMP_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise", format="mixed")
    for column in ("Pay_Date",):
        if column in frame:
            frame[column] = frame[column].replace("", pd.NA)
    for column in ("Open", "High", "Low", "Close", "Volume", "Adjusted_Close", "Split_Ratio",
                   "Amount_Per_Share", "Quoted_Rate", "Reference_Return", "Credited_Return"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column].replace("", np.nan), errors="raise")
    return _coerce(frame)


def _metadata_check(metadata):
    required = ["Track", "Instrument_ID", "Calendar", "Exchange_Timezone", "Source",
                "Request", "Price_Basis", "Availability_Assumptions", "Redistribution"]
    if any(key not in metadata or not metadata[key] for key in required):
        raise DataValidationError(f"Metadata require nonempty fields {required}")
    if "Retrieved_UTC" not in metadata or "Imported_UTC" not in metadata:
        raise DataValidationError("Metadata must distinguish retrieval time from import time")
    utc_timestamp(metadata["Imported_UTC"])
    if metadata["Retrieved_UTC"] is not None:
        utc_timestamp(metadata["Retrieved_UTC"])


def _check_instrument_metadata(bars, metadata):
    if bars.empty or not bars.Instrument_ID.eq(metadata["Instrument_ID"]).all():
        raise DataValidationError("Instrument metadata disagree with market bars")
    local_sessions = bars.Open_Time.dt.tz_convert(metadata["Exchange_Timezone"]).dt.strftime("%Y-%m-%d")
    if not local_sessions.eq(bars.Session).all():
        raise DataValidationError("Market opens disagree with exchange-local session dates")


def write_snapshot(*, bars, schedule, metadata, root="datasets", actions=None,
                   cash_observations=None, cash_policy=None, validation_policy=None,
                   raw_files=None) -> Dataset:
    """Create a new snapshot, or verify and reuse an identical existing one.

    Dataset_ID is inherited from the manifest and materialized on API tables.
    Backing CSV files omit it to avoid circular content hashes.
    """
    _metadata_check(metadata)
    cash_policy = cash_policy or CashPolicy()
    validation_policy = validation_policy or ValidationPolicy()
    tables = {
        "bars": _coerce(bars), "schedule": _coerce(schedule),
        "actions": _coerce(actions if actions is not None else empty_table("actions")),
        "cash_observations": _coerce(cash_observations if cash_observations is not None
                                    else empty_table("cash_observations")),
    }
    for name, frame in tables.items():
        missing = set(TABLE_COLUMNS[name]) - set(frame.columns)
        if missing:
            raise DataValidationError(f"{name} missing columns {sorted(missing)}")
    tables["cash_intervals"] = _coerce(build_cash_intervals(tables["bars"], tables["cash_observations"], cash_policy))
    _check_instrument_metadata(tables["bars"], metadata)
    report = validate_tables(tables, validation_policy)
    reconciliation = reconcile_total_returns(tables["bars"], tables["actions"])
    if not reconciliation.empty:
        residual = float(reconciliation.Adjustment_Residual.abs().max())
        report["Return_Reconciliation"] = {
            "Intervals": len(reconciliation), "Maximum_Adjustment_Residual": residual,
            "Tolerance": validation_policy.reconciliation_tolerance,
            "Maximum_Cash_Convention_Gap": float(reconciliation.Cash_Convention_Gap.abs().max()),
            "Convention": "provider_backward_dividend_multiplier_vs_ex_close_cash_reinvestment",
        }
        if not np.isfinite(residual) or residual > validation_policy.reconciliation_tolerance:
            raise DataValidationError(f"Provider total-return reconciliation failed, residual {residual}")
    files = {f"{name}.csv": _csv_bytes(frame) for name, frame in tables.items()}
    files["reconciliation.csv"] = _csv_bytes(reconciliation)
    files["quality_report.json"] = _json_bytes(report)
    for name, value in (raw_files or {}).items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."}:
            raise DataValidationError("Raw filenames must be simple names")
        if not isinstance(value, bytes):
            raise TypeError("Raw source files must be supplied as bytes")
        files[f"raw/{name}"] = value
    if not raw_files:
        raise DataValidationError("At least one original source file is required")
    payload = {
        "Schema_Version": SCHEMA_VERSION, "Metadata": metadata,
        "Cash_Policy": cash_policy.to_dict(), "Validation_Policy": asdict(validation_policy),
        "Runtime": {"Python": platform.python_version(), **{name: importlib.metadata.version(name)
                    for name in ["pandas", "numpy", "pandas-market-calendars"]}},
        "Data_Code_SHA256": {path.name: _digest(path.read_bytes())
                             for path in sorted(Path(__file__).parent.glob("*.py"))},
        "Files": {name: {"SHA256": _digest(value), "Bytes": len(value)}
                  for name, value in sorted(files.items())},
    }
    dataset_id = "sha256-" + _digest(_json_bytes(payload))
    manifest = {"Dataset_ID": dataset_id, **payload}
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / dataset_id
    if not destination.exists():
        with tempfile.TemporaryDirectory(prefix=".pending-", dir=root) as temporary:
            pending = Path(temporary) / dataset_id
            pending.mkdir()
            for name, value in files.items():
                path = pending / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(value)
            (pending / "manifest.json").write_bytes(_json_bytes(manifest))
            # Validate the serialized representation before publishing any snapshot.
            load_dataset(dataset_id, root=Path(temporary))
            try:
                pending.rename(destination)
            except FileExistsError:
                pass
    return load_dataset(dataset_id, root=root)


def load_dataset(dataset_id: str, root="datasets") -> Dataset:
    """Load only local bytes and reject a damaged, altered, or partial snapshot."""
    if not re.fullmatch(r"sha256-[0-9a-f]{64}", dataset_id):
        raise DataValidationError("Invalid dataset identifier")
    directory = Path(root) / dataset_id
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        payload = {key: value for key, value in manifest.items() if key != "Dataset_ID"}
        if (manifest["Dataset_ID"] != dataset_id or "sha256-" + _digest(_json_bytes(payload)) != dataset_id):
            raise DataValidationError("Dataset manifest does not match its content address")
        if manifest["Schema_Version"] != SCHEMA_VERSION:
            raise DataValidationError("Unsupported snapshot schema")
        _metadata_check(manifest["Metadata"])
        required = {f"{name}.csv" for name in TABLE_COLUMNS} | {"quality_report.json", "reconciliation.csv"}
        if not required.issubset(manifest["Files"]) or not any(n.startswith("raw/") for n in manifest["Files"]):
            raise DataValidationError("Snapshot lacks required files")
        for name, expected in manifest["Files"].items():
            parts = name.split("/")
            if (any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part) or part in {".", ".."} for part in parts)
                    or len(parts) > 2 or (len(parts) == 2 and parts[0] != "raw")):
                raise DataValidationError("Unsafe manifest path")
            path = directory.joinpath(*parts)
            if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                raise DataValidationError("Snapshot file escapes its directory")
            value = path.read_bytes()
            if len(value) != expected["Bytes"] or _digest(value) != expected["SHA256"]:
                raise DataValidationError(f"Checksum mismatch for {name}")
        tables = {name: _read_table(directory / f"{name}.csv") for name in TABLE_COLUMNS}
        _check_instrument_metadata(tables["bars"], manifest["Metadata"])
        validate_tables(tables, ValidationPolicy(**manifest["Validation_Policy"]))
        reconstructed = build_cash_intervals(tables["bars"], tables["cash_observations"],
                                             CashPolicy(**manifest["Cash_Policy"]))
        if not np.allclose(reconstructed[["Reference_Return", "Credited_Return"]].to_numpy(float),
                           tables["cash_intervals"][["Reference_Return", "Credited_Return"]].to_numpy(float),
                           rtol=1e-12, atol=1e-15):
            raise DataValidationError("Cash intervals do not reconcile to the saved cash model")
        if reconstructed.Observation_IDs.tolist() != tables["cash_intervals"].Observation_IDs.tolist():
            raise DataValidationError("Saved cash quote selection differs from the declared policy")
        report = json.loads((directory / "quality_report.json").read_text(encoding="utf-8"))
        reconciliation = pd.read_csv(directory / "reconciliation.csv", float_precision="round_trip")
        for frame in tables.values():
            frame.insert(0, "Dataset_ID", dataset_id)
        return Dataset(manifest=manifest, quality_report=report, reconciliation=reconciliation, **tables)
    except DataValidationError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise DataValidationError(f"Cannot load complete dataset {dataset_id}: {error}") from error

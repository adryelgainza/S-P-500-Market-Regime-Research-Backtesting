"""Chronology, stage separation, fixed transfer aggregation, and evidence gates."""

import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from .data import load_dataset
from .data.calendar import exchange_schedule
from .experiments import atomic_write, canonical_bytes, content_hash, utc_now
from .protocol import aware_time, verify_freeze
from .statistics import (annualized_sharpe, deflated_sharpe, initial_capital_drawdown,
                         max_t_adjustment, paired_sharpe_bootstrap)


VALIDITY_GATES = ("chronology", "accounting", "provenance", "reproducibility", "trial_disclosure", "test_calibration")


def _times(frame, names):
    if frame.empty or not set(names).issubset(frame):
        raise ValueError(f"Chronology audit requires nonempty columns {names}")
    return {name: pd.Series([aware_time(value) for value in frame[name]], index=frame.index) for name in names}


def validate_refit(base, calibration, refit_time):
    """Base targets mature before calibration, and all labels precede the refit."""
    refit = aware_time(refit_time)
    columns = ["Row_ID", "Decision_Time", "Available_At", "Label_End_Time"]
    if not set(columns).issubset(base) or not set(columns).issubset(calibration):
        raise ValueError("Refit audits require row IDs, availability, decision and label-end timestamps")
    all_rows = pd.concat([base, calibration], ignore_index=True)
    if all_rows.Row_ID.isna().any() or all_rows.Row_ID.duplicated().any():
        raise ValueError("Base and calibration rows must have disjoint unique IDs")
    times = _times(all_rows, columns[1:])
    if (times["Available_At"] > times["Decision_Time"]).any() or (times["Label_End_Time"] <= times["Decision_Time"]).any():
        raise ValueError("Features arrived after their decision or targets precede their feature rows")
    start = _times(calibration, ["Decision_Time"])["Decision_Time"].min()
    base_times = _times(base, columns[1:])
    cal_times = _times(calibration, columns[1:])
    if ((base_times["Label_End_Time"] >= start).any() or (cal_times["Label_End_Time"] >= refit).any()
            or (times["Available_At"] > refit).any() or start >= refit):
        raise ValueError("Immature training or calibration labels cross a fit boundary")
    return {"Base_Rows": len(base), "Calibration_Rows": len(calibration), "Refit_UTC": refit.isoformat(),
            "Audit_SHA256": content_hash(all_rows.astype(str).to_dict("records"))}


def validate_selection(returns, decision_time):
    times = _times(returns, ["Return_End_Time"])["Return_End_Time"]
    if (times >= aware_time(decision_time)).any():
        raise ValueError("Selection used returns ending at or after its decision")
    return {"Rows": len(returns), "Latest_Return_End": times.max().isoformat()}


def validate_cash_timing(cash):
    times = _times(cash, ["Available_At", "Interval_Start"])
    if (times["Available_At"] > times["Interval_Start"]).any():
        raise ValueError("Cash rate was unavailable when accrual began")
    return {"Intervals": len(cash)}


def check_prefix_invariance(raw, decision_function, decisions):
    """Rebuild each decision from raw prefixes through the normal callable entry point."""
    available = _times(raw, ["Available_At"])["Available_At"]
    records = []
    for value in decisions:
        timestamp = aware_time(value)
        prefix = raw.loc[available <= timestamp].copy(deep=True)
        full_result = decision_function(raw.copy(deep=True), timestamp)
        prefix_result = decision_function(prefix, timestamp)
        if content_hash(full_result) != content_hash(prefix_result):
            raise ValueError(f"Future data changed a decision at {timestamp.isoformat()}")
        records.append({"Decision_UTC": timestamp.isoformat(), "Decision_SHA256": content_hash(prefix_result)})
    if not records:
        raise ValueError("Prefix audit requires at least one decision")
    return {"Checked_Decisions": records, "Raw_SHA256": content_hash(raw.astype(str).to_dict("records"))}


def aggregate_transfer(returns, protocol):
    """Equal-calendar intersection is explicit. Missing markets or dates are rejected."""
    weights = {row["Instrument"]: row["Weight"] for row in protocol["Universe"]["Transfer"]}
    if set(returns) != set(weights):
        raise ValueError("Every registered transfer market must be reported")
    first = returns[next(iter(weights))]
    if first.empty or first.index.has_duplicates or not first.index.is_monotonic_increasing:
        raise ValueError("Transfer dates must be nonempty, unique and ordered")
    result = first*0
    for instrument, weight in weights.items():
        frame = returns[instrument]
        if not frame.index.equals(first.index) or not frame.columns.equals(first.columns):
            raise ValueError("Transfer aggregation requires the predetermined identical date and strategy mask")
        if not np.isfinite(frame.to_numpy(float)).all():
            raise ValueError("Missing transfer values require an incomplete evaluation")
        result = result+weight*frame
    return result


def validate_run_binding(specification, verified, instrument):
    """One scientific configuration governs every market and every repeat attempt."""
    protocol, body = verified["Protocol"], verified["Receipt"]["Body"]
    expected = content_hash({key: protocol[key] for key in ["Models", "Allocation"]})
    if (specification["Hypothesis_ID"] != protocol["Hypothesis"]["ID"]
            or specification["Configuration_SHA256"] != expected
            or specification["Source_Revision"] != body["Source_Revision"]
            or specification["Seed"] != protocol["Models"]["Seed"]
            or specification.get("Protocol_SHA256") != body["Protocol_SHA256"]):
        raise ValueError("Run differs from its frozen hypothesis, recipe, seed or source")
    stage = specification["Stage"]
    allowed = {row["Instrument"] for row in protocol["Universe"]["Transfer"]} if stage == "transfer" else {protocol["Universe"]["Primary"]}
    if instrument not in allowed:
        raise ValueError("Run instrument differs from its registered evaluation stage")
    return {"Stage": stage, "Instrument": instrument, "Configuration_SHA256": expected}


def _decision_binding(decision, protocol, body):
    required = {"Protocol_SHA256", "Configuration_SHA256", "Source_Revision", "Dataset_ID", "Instrument",
                "Candidate_ID", "Decision_Time", "Execution_Time", "Inputs_Available_At", "Target_Weight", "Artifacts"}
    if not required.issubset(decision):
        raise ValueError("Decision requires complete frozen provenance and artifact references")
    if (decision["Protocol_SHA256"] != body["Protocol_SHA256"]
            or decision["Configuration_SHA256"] != content_hash({key: protocol[key] for key in ["Models", "Allocation"]})
            or decision["Source_Revision"] != body["Source_Revision"]
            or decision["Candidate_ID"] not in protocol["Allocation"]["Candidates"]
            or decision["Instrument"] != protocol["Universe"]["Primary"]):
        raise ValueError("Decision differs from its frozen rule or primary forward universe")
    timestamp, execution = aware_time(decision["Decision_Time"]), aware_time(decision["Execution_Time"])
    if timestamp < aware_time(body["First_Scheduled_Decision_UTC"]) or aware_time(decision["Inputs_Available_At"]) > timestamp:
        raise ValueError("Forward decision precedes the freeze or uses late inputs")
    if execution <= timestamp or not 0 <= decision["Target_Weight"] <= 1:
        raise ValueError("Decision requires a later execution and a long/cash weight")
    schedule = exchange_schedule(timestamp.date(), (timestamp+pd.Timedelta(days=14)).date(), protocol["Universe"]["Calendar"])
    matching = schedule.loc[schedule.Close_Time+pd.Timedelta(minutes=protocol["Evaluation"]["Decision_Delay_Minutes"]) == timestamp]
    following = schedule.loc[schedule.Open_Time > timestamp]
    if len(matching) != 1 or following.empty or following.iloc[0].Open_Time != execution:
        raise ValueError("Decision and execution must use the registered close and next exchange open")
    return timestamp, execution


def record_forward_decision(project, frozen_directory, store, experiment_id, decision):
    """Persist a signal after inputs arrive and before its scheduled execution."""
    verified = verify_freeze(frozen_directory, store, project=project)
    protocol, body = verified["Protocol"], verified["Receipt"]["Body"]
    timestamp, execution = _decision_binding(decision, protocol, body)
    dataset = load_dataset(decision["Dataset_ID"], root=Path(project)/"datasets")
    last = dataset.bars.loc[dataset.bars.Available_At <= timestamp]
    if (last.empty or str(last.iloc[-1].Instrument_ID) != decision["Instrument"]
            or aware_time(last.iloc[-1].Available_At) != timestamp):
        raise ValueError("Forward snapshot lacks the current completed session")
    if not decision["Artifacts"]:
        raise ValueError("Forward decisions require saved score and chronology artifacts")
    for artifact in decision["Artifacts"]:
        path = (Path(project)/artifact["Path"]).resolve()
        if not path.is_relative_to(Path(project).resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != artifact["SHA256"]:
            raise ValueError("Decision artifact hash mismatch")
    with store.transaction():
        events, states, _ = store._read()
        if experiment_id not in states or states[experiment_id]["State"] != "started":
            raise ValueError("Forward decisions require a started registered attempt")
        spec = states[experiment_id]["Specification"]
        validate_run_binding(spec, verified, decision["Instrument"])
        if (spec["Stage"] != "forward" or spec["Hypothesis_ID"] != protocol["Hypothesis"]["ID"]
                or spec["Configuration_SHA256"] != decision["Configuration_SHA256"]
                or spec["Source_Revision"] != body["Source_Revision"] or decision["Dataset_ID"] not in spec["Dataset_IDs"]):
            raise ValueError("Forward attempt provenance differs from the decision")
        key = f"decision-{body['Protocol_SHA256']}-{decision['Instrument']}-{timestamp.isoformat()}"
        existing = [event for event in events if event["Idempotency_Key"] == key]
        if existing:
            if existing[0]["Payload"] != decision or existing[0]["Experiment_ID"] != experiment_id:
                raise ValueError("A committed forward decision cannot be replaced")
            return existing[0]
        now = aware_time(utc_now())
        if not timestamp <= now < execution:
            raise ValueError("Forward decisions must be recorded after the decision and before execution")
        return store._append_many([{"Type": "forward_decision_recorded", "Experiment_ID": experiment_id,
            "Idempotency_Key": key, "Recorded_UTC": now.isoformat(), "Payload": decision}])[0]


def validate_stage(stage, instrument, decisions, verified, store):
    """Bind evaluated decisions to their real stage and any contemporaneous records."""
    protocol, body = verified["Protocol"], verified["Receipt"]["Body"]
    times = _times(decisions, ["Decision_Time", "Execution_Time", "Return_End_Time"])
    if times["Decision_Time"].duplicated().any() or not times["Decision_Time"].is_monotonic_increasing:
        raise ValueError("Evaluation decisions must be unique and ordered")
    if (times["Execution_Time"] <= times["Decision_Time"]).any() or (times["Return_End_Time"] <= times["Execution_Time"]).any():
        raise ValueError("Evaluation timeline is reversed")
    if (times["Return_End_Time"] > aware_time(utc_now())).any():
        raise ValueError("Evaluation contains unrealized returns")
    if stage in {"retrospective", "development"}:
        return {"Stage": stage, "Fresh_Time_Evidence": False}
    if stage == "transfer":
        markets = {row["Instrument"]:row for row in protocol["Universe"]["Transfer"]}
        if instrument not in markets:
            raise ValueError("Unregistered transfer market")
        start, end = protocol["Evaluation"]["Transfer"]["Start"], protocol["Evaluation"]["Transfer"]["End_Exclusive"]
        dates = times["Decision_Time"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        if (dates < start).any() or (dates >= end).any():
            raise ValueError("Transfer dates lie outside the registered window")
        schedule = exchange_schedule(start, end, protocol["Universe"]["Calendar"])
        expected = schedule.loc[schedule.Session < end, "Close_Time"]+pd.Timedelta(minutes=protocol["Evaluation"]["Decision_Delay_Minutes"])
        # Every session in the registered window is required for a complete transfer release.
        if list(times["Decision_Time"]) != list(expected):
            raise ValueError("Transfer evaluation omits registered sessions")
        return {"Stage": stage, "Fresh_Time_Evidence": False, "Previously_Inspected": markets[instrument]["Previously_Inspected"]}
    if stage != "forward" or instrument != protocol["Universe"]["Primary"]:
        raise ValueError("Invalid evaluation stage or forward instrument")
    if (times["Decision_Time"] < aware_time(body["First_Scheduled_Decision_UTC"])).any():
        raise ValueError("Historical observations cannot be assigned to the forward stage")
    first = aware_time(body["First_Scheduled_Decision_UTC"])
    schedule = exchange_schedule(first.date(), times["Decision_Time"].max().date(), protocol["Universe"]["Calendar"])
    expected = schedule.Close_Time+pd.Timedelta(minutes=protocol["Evaluation"]["Decision_Delay_Minutes"])
    expected = expected[expected >= first]
    if list(times["Decision_Time"]) != list(expected):
        raise ValueError("Forward evaluation omits scheduled decisions and is incomplete")
    with store.transaction():
        events, _, _ = store._read(verified["Receipt"]["Anchor"])
    records = {aware_time(event["Payload"]["Decision_Time"]):event for event in events
               if event["Type"] == "forward_decision_recorded" and event["Payload"]["Protocol_SHA256"] == body["Protocol_SHA256"]}
    for row in decisions.to_dict("records"):
        timestamp = aware_time(row["Decision_Time"])
        event = records.get(timestamp)
        if event is None or aware_time(event["Recorded_UTC"]) >= aware_time(row["Execution_Time"]):
            raise ValueError("Forward evaluation lacks a contemporaneous committed decision")
        _decision_binding(event["Payload"], protocol, body)
        if (aware_time(event["Payload"]["Execution_Time"]) != aware_time(row["Execution_Time"])
                or row.get("Target_Weight") != event["Payload"]["Target_Weight"]):
            raise ValueError("Evaluated forward targets differ from the committed decisions")
    return {"Stage": stage, "Fresh_Time_Evidence": True, "Committed_Decisions": len(decisions)}


def assess_gates(validity, metrics, protocol):
    """Keep methodological validity independent from evidence of incremental value."""
    valid = {name: validity.get(name, "inconclusive") for name in VALIDITY_GATES}
    if any(value not in {"passed", "failed", "inconclusive"} for value in valid.values()):
        raise ValueError("Gate states must be passed, failed or inconclusive")
    thresholds = protocol["Inference"]["Thresholds"]
    adequate = ((metrics.get("Observations") or 0) >= thresholds["Minimum_Observations"]
                and (metrics.get("Episodes") or 0) >= thresholds["Minimum_Episodes"])
    tests = {"incremental_sharpe": ("Delta_Sharpe", thresholds["Minimum_Delta_Sharpe"], "ge"),
             "uncertainty": ("Delta_Sharpe_Lower", thresholds["CI_Lower_Min"], "gt"),
             "multiplicity": ("Adjusted_P", thresholds["Adjusted_P_Max"], "le"),
             "deflated_sharpe": ("DSR", thresholds["DSR_Min"], "ge"),
             "risk": ("Drawdown_Magnitude", thresholds["Maximum_Drawdown"], "le"),
             "cost_sensitivity": ("Cost_Delta_Sharpe", thresholds["Cost_Delta_Sharpe_Min"], "ge")}
    evidence = {"sample": "passed" if adequate else "inconclusive"}
    for name, (key, threshold, direction) in tests.items():
        value = metrics.get(key)
        if not adequate or value is None or not np.isfinite(value):
            evidence[name] = "inconclusive"
        else:
            passed = {"ge": value >= threshold, "gt": value > threshold, "le": value <= threshold}[direction]
            evidence[name] = "passed" if passed else "failed"
    def overall(values):
        if "failed" in values:
            return "failed"
        return "passed" if all(value == "passed" for value in values) else "inconclusive"
    validity_state, evidence_state = overall(valid.values()), overall(evidence.values())
    return {"Research_Validity": valid, "Research_Validity_Status": validity_state,
            "Strategy_Evidence": evidence, "Strategy_Evidence_Status": evidence_state,
            "Promotion": "eligible" if validity_state == evidence_state == "passed" else "withheld"}


def evaluate_fixed_returns(portfolio, control, targets, cost_portfolio, cost_control, protocol):
    """Compute frozen evidence metrics from aligned net/excess streams and target weights."""
    series = [portfolio, control, cost_portfolio, cost_control]
    if any(not value.index.equals(portfolio.index) for value in [*series, targets]):
        raise ValueError("Evidence inputs must share an identical chronological date mask")
    if portfolio.index.has_duplicates or not portfolio.index.is_monotonic_increasing:
        raise ValueError("Evidence dates must be unique and ordered")
    if any(not {"Net_Return", "Net_Excess_Return"}.issubset(value) for value in series):
        raise ValueError("Evidence requires net total and excess returns at both cost levels")
    if any(not np.isfinite(value.to_numpy(float)).all() or (value.Net_Return <= -1).any() for value in series):
        raise ValueError("Evidence returns must be finite and preserve positive wealth")
    if not targets.between(0, 1).all():
        raise ValueError("Episode targets exceed the registered portfolio bounds")
    active = targets > 1e-12
    episodes = int((active & ~active.shift(fill_value=False)).sum())
    metrics = {"Observations": len(portfolio), "Episodes": episodes}
    threshold = protocol["Inference"]["Thresholds"]
    if len(portfolio) > protocol["Evaluation"]["Forward"]["Minimum_Completed_Intervals"]:
        raise ValueError("Evidence exceeds the frozen review horizon. Register an extension")
    if len(portfolio) >= 2:
        metrics.update(Delta_Sharpe=float(annualized_sharpe(portfolio.Net_Excess_Return)-annualized_sharpe(control.Net_Excess_Return)),
            Drawdown_Magnitude=float(-initial_capital_drawdown(np.cumprod(1+portfolio.Net_Return.to_numpy())).min()),
            Cost_Delta_Sharpe=float(annualized_sharpe(cost_portfolio.Net_Excess_Return)-annualized_sharpe(cost_control.Net_Excess_Return)))
        metrics = {key: None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value for key, value in metrics.items()}
    if len(portfolio) < threshold["Minimum_Observations"] or episodes < threshold["Minimum_Episodes"]:
        return metrics
    config = protocol["Inference"]
    candidate, benchmark = portfolio.Net_Excess_Return, control.Net_Excess_Return
    bootstrap = paired_sharpe_bootstrap(candidate, benchmark, repetitions=config["Uncertainty"]["Replications"],
        block_length=config["Uncertainty"]["Block_Length"], seed=protocol["Models"]["Seed"], confidence_level=config["Confidence"])
    family = pd.DataFrame({config["Family"][0]: candidate}, index=portfolio.index)
    max_t = max_t_adjustment(family, benchmark, repetitions=config["Multiplicity"]["Replications"],
        block_length=config["Uncertainty"]["Block_Length"], maximum_lag=config["Multiplicity"]["HAC_Lag"], seed=protocol["Models"]["Seed"])
    dsr, _, _, _ = deflated_sharpe(candidate, family, 1)
    raw = dsr.loc[dsr.Scenario.eq("raw_registered_count")].iloc[0]
    equity = np.cumprod(1+portfolio.Net_Return.to_numpy())
    metrics.update(Delta_Sharpe=float(annualized_sharpe(candidate)-annualized_sharpe(benchmark)),
        Delta_Sharpe_Lower=bootstrap["Delta_Sharpe_Lower"], Adjusted_P=float(max_t.iloc[0].MaxT_Adjusted_P), DSR=raw.DSR,
        Drawdown_Magnitude=float(-initial_capital_drawdown(equity).min()),
        Cost_Delta_Sharpe=float(annualized_sharpe(cost_portfolio.Net_Excess_Return)-annualized_sharpe(cost_control.Net_Excess_Return)))
    return {key: None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value for key, value in metrics.items()}


def record_evaluation(project, frozen_directory, store, experiment_id, instrument, decisions, portfolio, control,
                      targets, cost_portfolio, cost_control, *, validity, artifacts):
    """Bind an audit-backed evaluation to its registered attempt and saved return inputs.

    Validity states are audit inputs from the accounting, fit, prefix, and runner
    verifiers. Missing audits remain inconclusive. Metric computation is independent
    of those states, so a valid negative result is retained.
    """
    project = Path(project).resolve()
    verified = verify_freeze(frozen_directory, store, project=project)
    protocol, body = verified["Protocol"], verified["Receipt"]["Body"]
    with store.transaction():
        _, states, _ = store._read(verified["Receipt"]["Anchor"])
        if experiment_id not in states or states[experiment_id]["State"] != "started":
            raise ValueError("Evaluations require a started registered attempt")
        spec = states[experiment_id]["Specification"]
    validate_run_binding(spec, verified, instrument)
    stage = validate_stage(spec["Stage"], instrument, decisions, verified, store)
    if not pd.DatetimeIndex(pd.to_datetime(decisions.Return_End_Time, utc=True)).equals(portfolio.index):
        raise ValueError("Evaluation decision rows differ from the supplied return mask")
    if not np.array_equal(decisions.Target_Weight.to_numpy(float), targets.to_numpy(float)):
        raise ValueError("Evidence episode weights differ from evaluated decisions")
    if any(validity.get(name) == "passed" for name in VALIDITY_GATES) and not artifacts:
        raise ValueError("Passing validity attestations require saved audit artifacts")
    for artifact in artifacts:
        path = (Path(project)/artifact["Path"]).resolve()
        if not path.is_relative_to(Path(project).resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != artifact["SHA256"]:
            raise ValueError("Evaluation audit artifact hash mismatch")
    metrics = evaluate_fixed_returns(portfolio, control, targets, cost_portfolio, cost_control, protocol)
    gates = assess_gates(validity, metrics, protocol)
    if spec["Stage"] != "forward":
        gates["Promotion"] = "withheld"
    identifier = uuid.uuid4().hex
    directory = store.directory.resolve()/"evaluations"/identifier
    if not directory.is_relative_to(project):
        raise ValueError("Evaluation archives must lie inside the project")
    directory.mkdir(parents=True)
    frames = {"decisions": decisions, "portfolio": portfolio, "control": control,
              "cost_portfolio": cost_portfolio, "cost_control": cost_control, "targets": targets.to_frame("Target_Weight")}
    references = list(artifacts)
    for name, frame in frames.items():
        path = directory/f"{name}.parquet"
        frame.to_parquet(path)
        references.append({"Path": path.relative_to(Path(project)).as_posix(), "SHA256": hashlib.sha256(path.read_bytes()).hexdigest()})
    payload = {"Protocol_SHA256": body["Protocol_SHA256"], "Stage": spec["Stage"], "Instrument": instrument,
               "Dataset_IDs": spec["Dataset_IDs"], "Metrics": metrics, "Validity_Attestations": validity,
               "Gates": gates, "Stage_Audit": stage, "Artifacts": references}
    atomic_write(directory/"evaluation.json", canonical_bytes(payload)+b"\n")
    payload["Artifacts"].append({"Path": (directory/"evaluation.json").relative_to(Path(project)).as_posix(),
                                  "SHA256": hashlib.sha256((directory/"evaluation.json").read_bytes()).hexdigest()})
    with store.transaction():
        _, states, _ = store._read()
        if states[experiment_id]["State"] != "started":
            raise ValueError("Attempt ended before evaluation commit")
        event = store._append_many([{"Type": "evaluation_recorded", "Experiment_ID": experiment_id,
            "Idempotency_Key": "evaluation-"+identifier, "Payload": payload}])[0]
    atomic_write(directory/"completion.json", canonical_bytes({"Event_SHA256": event["Event_SHA256"],
                 "Sequence": event["Sequence"], "Protocol_SHA256": body["Protocol_SHA256"]})+b"\n")
    return payload


def verify_evaluation(project, event, verified, store):
    """Recompute saved evidence and gates from event-bound return inputs."""
    project = Path(project).resolve()
    payload = event["Payload"]
    for artifact in payload["Artifacts"]:
        path = (project/artifact["Path"]).resolve()
        if not path.is_relative_to(project) or hashlib.sha256(path.read_bytes()).hexdigest() != artifact["SHA256"]:
            raise ValueError("Evaluation artifact changed")
    saved = [artifact for artifact in payload["Artifacts"] if Path(artifact["Path"]).name == "evaluation.json"]
    if len(saved) != 1:
        raise ValueError("Evaluation needs a uniquely bound snapshot")
    directory = (project/saved[0]["Path"]).parent
    expected = dict(payload, Artifacts=[artifact for artifact in payload["Artifacts"] if artifact != saved[0]])
    if json.loads((directory/"evaluation.json").read_text(encoding="utf-8")) != expected:
        raise ValueError("Evaluation summary differs from its committed event")
    frames = {name: pd.read_parquet(directory/f"{name}.parquet") for name in
              ["decisions", "portfolio", "control", "targets", "cost_portfolio", "cost_control"]}
    with store.transaction():
        events, states, _ = store._read(verified["Receipt"]["Anchor"])
    if not any(value["Event_SHA256"] == event["Event_SHA256"] for value in events):
        raise ValueError("Evaluation event is absent from the experiment history")
    spec = states[event["Experiment_ID"]]["Specification"]
    validate_run_binding(spec, verified, payload["Instrument"])
    stage = validate_stage(spec["Stage"], payload["Instrument"], frames["decisions"], verified, store)
    metrics = evaluate_fixed_returns(frames["portfolio"], frames["control"], frames["targets"].Target_Weight,
        frames["cost_portfolio"], frames["cost_control"], verified["Protocol"])
    gates = assess_gates(payload["Validity_Attestations"], metrics, verified["Protocol"])
    if spec["Stage"] != "forward":
        gates["Promotion"] = "withheld"
    if content_hash(metrics) != content_hash(payload["Metrics"]) or gates != payload["Gates"] or stage != payload["Stage_Audit"]:
        raise ValueError("Evaluation metrics, stage or gates fail independent recomputation")
    return {"Verified": True, "Stage": spec["Stage"], "Observations": metrics["Observations"]}

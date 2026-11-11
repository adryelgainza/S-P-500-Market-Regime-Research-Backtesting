"""Derive eligible dates from real history and timestamp-based label purging."""

import numpy as np
import pandas as pd

from .contracts import Dataset, EligibleSample, HistoryRequirements


def build_eligible_sample(dataset: Dataset, history_requirements: HistoryRequirements) -> EligibleSample:
    requirements = history_requirements
    bars = dataset.bars.reset_index(drop=True)
    sessions = pd.DatetimeIndex(bars.Session)
    opens = pd.DatetimeIndex(bars.Open_Time)
    closes = pd.DatetimeIndex(bars.Close_Time)
    decisions = closes + pd.Timedelta(minutes=requirements.decision_delay_minutes)
    first_feature = requirements.feature_warmup_sessions - 1
    label_shift = requirements.label_horizon_sessions + 1
    rows, exclusions = [], []
    # Strict-calendar datasets make a missing row impossible. For permissive
    # historical snapshots, gaps disqualify any window that spans the gap.
    expected = list(dataset.schedule.Session)
    expected_positions = {session: i for i, session in enumerate(expected)}
    positions = np.array([expected_positions.get(s, -1) for s in bars.Session])
    gap = np.r_[False, np.diff(positions) != 1]
    gap_count = np.cumsum(gap)
    unavailable_features = (pd.DatetimeIndex(bars.Available_At) > decisions).astype(int)
    unavailable_prefix = np.r_[0, np.cumsum(unavailable_features)]
    for index in range(len(bars)):
        reasons = []
        window_start = sessions[index] - pd.DateOffset(years=requirements.training_years)
        start = int(sessions.searchsorted(window_start, side="left"))
        last_mature = index - label_shift
        calibration_start = last_mature - requirements.calibration_sessions + 1
        if requirements.calibration_sessions:
            training_end = calibration_start - label_shift
        else:
            calibration_start = None
            training_end = last_mature
        effective_start = max(start, first_feature)
        if index < first_feature:
            reasons.append("feature_warmup")
        if first_feature >= len(bars) or sessions[min(first_feature, len(bars)-1)] > window_start:
            reasons.append("insufficient_calendar_history")
        if training_end - effective_start + 1 < requirements.minimum_training_rows:
            reasons.append("insufficient_purged_training_rows")
        if calibration_start is not None and calibration_start < max(start, first_feature):
            reasons.append("insufficient_mature_calibration_rows")
        execution = index + 1
        return_end = execution + requirements.holding_sessions
        if return_end >= len(bars):
            reasons.append("unrealized_forward_return")
        coverage_end = min(return_end, len(bars)-1)
        coverage_start = max(0, effective_start - requirements.feature_warmup_sessions + 1)
        if (positions[index] < 0 or (coverage_end >= coverage_start
                and gap_count[coverage_end] - gap_count[coverage_start] > 0)):
            reasons.append("missing_exchange_session_in_required_window")
        if bars.Available_At.iloc[index] > decisions[index]:
            reasons.append("bar_not_available_at_decision")
        if unavailable_prefix[index + 1] - unavailable_prefix[coverage_start] > 0:
            reasons.append("unavailable_feature_rows_in_required_window")
        if reasons:
            exclusions.append({"Dataset_ID": dataset.dataset_id, "Session": bars.Session.iloc[index],
                               "Decision_Time": decisions[index], "Reasons": "|".join(reasons)})
            continue
        row = dict(Dataset_ID=dataset.dataset_id, Session=bars.Session.iloc[index],
                   Decision_Time=decisions[index], Execution_Time=opens[execution],
                   Return_End_Time=opens[return_end], Training_Start=decisions[effective_start],
                   Training_End=decisions[training_end], Training_Rows=training_end-effective_start+1,
                   Training_Label_End=opens[training_end + label_shift])
        if calibration_start is not None:
            row.update(Calibration_Start=decisions[calibration_start],
                       Calibration_End=decisions[last_mature], Calibration_Label_End=opens[index])
        rows.append(row)
    eligible_columns = ["Dataset_ID", "Session", "Decision_Time", "Execution_Time", "Return_End_Time",
                        "Training_Start", "Training_End", "Training_Rows", "Training_Label_End"]
    if requirements.calibration_sessions:
        eligible_columns += ["Calibration_Start", "Calibration_End", "Calibration_Label_End"]
    return EligibleSample(pd.DataFrame(rows, columns=eligible_columns),
                          pd.DataFrame(exclusions, columns=["Dataset_ID", "Session", "Decision_Time", "Reasons"]))

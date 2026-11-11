"""Regressions for Layer 5 experiment integrity, freezes, chronology and gates."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from regime_research.experiments import ExperimentLogError, ExperimentStore, content_hash
from regime_research.protocol import freeze_protocol, validate_protocol, verify_freeze
from regime_research.validation import (VALIDITY_GATES, aggregate_transfer, assess_gates, check_prefix_invariance,
    evaluate_fixed_returns, record_evaluation, record_forward_decision, validate_cash_timing, validate_refit, validate_run_binding,
    validate_selection, validate_stage, verify_evaluation)
from regime_research.validation_research import comparison_table


PROJECT = Path(__file__).resolve().parents[2]


def protocol():
    return json.loads((PROJECT/"protocols/layer5_validation_v1.json").read_text())


def specification(hypothesis="idea-1", parent=None):
    return {"Hypothesis_ID": hypothesis, "Configuration_SHA256": "a"*64, "Source_Revision": "sha256-"+"b"*64,
            "Dataset_IDs": ["sha256-"+"c"*64], "Parent_Experiment_ID": parent,
            "Stage": "development", "Seed": 42, "Purpose": "regression"}


def register(store, identifier, spec=None):
    return store.append("experiment_registered", identifier, {"Specification": spec or specification()},
                        idempotency_key="register-"+identifier)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        base = PROJECT/"tmp/validation_tests"
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = TemporaryDirectory(dir=base)
        self.addCleanup(self.temporary.cleanup)
        self.store = ExperimentStore(Path(self.temporary.name)/"experiments")

    def test_append_preserves_prefix_and_retry_is_idempotent(self):
        first = register(self.store, "one")
        raw = self.store.path.read_bytes()
        self.assertEqual(register(self.store, "one"), first)
        register(self.store, "two", specification(parent="one"))
        self.assertTrue(self.store.path.read_bytes().startswith(raw))
        self.assertEqual(self.store.verify()["Experiments"], 2)

    def test_reused_key_cannot_change_payload(self):
        register(self.store, "one")
        with self.assertRaises(ExperimentLogError):
            register(self.store, "one", specification("different"))

    def test_illegal_transition_and_missing_parent(self):
        register(self.store, "one")
        with self.assertRaises(ExperimentLogError):
            self.store.append("experiment_completed", "one", {}, idempotency_key="finish")
        with self.assertRaises(ExperimentLogError):
            register(self.store, "two", specification(parent="unknown"))
        self.assertEqual(self.store.verify()["Events"], 1)

    def test_atomic_batch_rejects_all_if_any_event_fails(self):
        events = [{"Type": "experiment_registered", "Experiment_ID": "one", "Payload": {"Specification": specification()},
                   "Idempotency_Key": "one"},
                  {"Type": "experiment_completed", "Experiment_ID": "one", "Payload": {}, "Idempotency_Key": "two"}]
        with self.assertRaises(ExperimentLogError):
            self.store.append_many(events)
        self.assertEqual(self.store.verify()["Events"], 0)

    def test_failures_abandoned_and_repeated_hypothesis_remain_visible(self):
        with self.assertRaisesRegex(RuntimeError, "fit failed"):
            with self.store.attempt(specification(), "failed"):
                raise RuntimeError("fit failed")
        with self.store.attempt(specification(parent="failed"), "repeat") as (_, outcome):
            outcome["Result"] = "negative"
        register(self.store, "abandoned")
        self.store.append("experiment_abandoned", "abandoned", {"Reason": "superseded before execution"}, idempotency_key="abandon")
        table = comparison_table(self.store.verify()["States"])
        self.assertEqual(set(table.State), {"failed", "completed", "abandoned"})
        self.assertEqual(table.Hypothesis_ID.nunique(), 1)
        self.assertIn("fit failed", table.loc[table.State.eq("failed"), "Failure_Reason"].iloc[0])
        with self.assertRaises(ExperimentLogError):
            with self.store.attempt(specification(), "repeat"):
                self.fail("Completed attempts cannot execute again under the same ID")

    def test_corrections_are_new_events_and_target_the_same_experiment(self):
        original = register(self.store, "one")
        register(self.store, "two")
        payload = {"Corrects_Event_ID": original["Event_ID"], "Reason": "corrected descriptive note", "Replacement": "new note"}
        with self.assertRaises(ExperimentLogError):
            self.store.append("experiment_corrected", "two", payload, idempotency_key="bad")
        self.store.append("experiment_corrected", "one", payload, idempotency_key="correction")
        self.assertEqual(len(self.store.verify()["States"]["one"]["Corrections"]), 1)

    def test_concurrent_writers_serialize(self):
        with ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(lambda i: register(self.store, str(i)), range(20)))
        self.assertEqual(self.store.verify()["Events"], 20)

    def test_tampering_partial_line_and_anchored_truncation_are_detected(self):
        register(self.store, "one")
        first = self.store.path.read_bytes()
        register(self.store, "two")
        original = self.store.path.read_bytes()
        anchor = self.store.save_anchor()
        for corrupted in [original.replace(b"idea-1", b"idea-2"), original[:-1], first]:
            self.store.path.write_bytes(corrupted)
            with self.assertRaises(ExperimentLogError):
                self.store.verify(anchor)

    def test_compare_and_swap_rejects_stale_head(self):
        anchor = self.store.save_anchor()
        register(self.store, "one")
        with self.assertRaises(ExperimentLogError):
            self.store.append("evaluation_recorded", "one", {}, idempotency_key="stale", expected_head=anchor["Event_SHA256"])


class FreezeTests(unittest.TestCase):
    def setUp(self):
        base = PROJECT/"tmp/validation_tests"
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = TemporaryDirectory(dir=base)
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        (self.project/"model.py").write_text("value = 1\n")
        self.document = protocol()
        self.document["Data"]["Dataset_IDs"] = ["sha256-"+"c"*64]
        self.path = self.project/"protocol.json"
        self.path.write_text(json.dumps(self.document))
        dataset = self.project/"datasets"/self.document["Data"]["Dataset_IDs"][0]
        dataset.mkdir(parents=True)
        (dataset/"manifest.json").write_text("{}")
        self.store = ExperimentStore(self.project/"experiments")
        self.addCleanup(patch.stopall)
        patch("regime_research.protocol.load_dataset").start()
        patch("regime_research.protocol.runtime_snapshot", return_value=({"Python": "3", "Platform": "test"}, b"numpy==2\n")).start()
        patch("regime_research.protocol.utc_now", return_value="2026-11-09T22:00:00+00:00").start()

    def freeze(self):
        return freeze_protocol(self.project, self.path, self.store)

    def test_freeze_receipt_binds_all_inputs_and_future_calendar(self):
        directory = self.freeze()
        verified = verify_freeze(directory, self.store, project=self.project)
        body = verified["Receipt"]["Body"]
        self.assertEqual(body["First_Scheduled_Decision_UTC"], "2026-11-10T21:01:00+00:00")
        self.assertIn("requirements.lock", body["Files"])
        self.assertIn("source/model.py", body["Source_SHA256"])
        self.assertEqual(self.freeze(), directory)
        self.assertEqual(self.store.verify()["Events"], 1)

    def test_frozen_changes_require_new_hash_and_version(self):
        original = self.freeze()
        self.document["Hypothesis"]["Predicted_Behavior"] = "a revised prediction"
        self.path.write_text(json.dumps(self.document))
        with self.assertRaisesRegex(ValueError, "new version"):
            self.freeze()
        self.document["Version"] = "1.0.1"
        self.path.write_text(json.dumps(self.document))
        new = self.freeze()
        one, two = [json.loads((path/"receipt.json").read_text())["Body"]["Protocol_SHA256"] for path in [original, new]]
        self.assertNotEqual(one, two)

    def test_snapshot_mutation_and_live_source_mutation_are_detected(self):
        directory = self.freeze()
        (self.project/"model.py").write_text("value = 2\n")
        with self.assertRaisesRegex(ValueError, "source changed"):
            verify_freeze(directory, self.store, project=self.project)
        verify_freeze(directory, self.store)
        (directory/"protocol.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            verify_freeze(directory, self.store)

    def test_committed_freeze_recovers_after_publication_interruption(self):
        with patch("regime_research.experiments.os.rename", side_effect=OSError("interrupted publication")):
            with self.assertRaises(OSError):
                self.freeze()
        directory = self.freeze()
        verify_freeze(directory, self.store)
        self.assertEqual(self.store.verify()["Events"], 1)

    def decision(self, verified):
        body = verified["Receipt"]["Body"]
        artifact = self.project/"scores.json"
        artifact.write_text('{"score":0.2}')
        return {"Protocol_SHA256": body["Protocol_SHA256"],
            "Configuration_SHA256": content_hash({key: self.document[key] for key in ["Models", "Allocation"]}),
            "Source_Revision": body["Source_Revision"], "Dataset_ID": self.document["Data"]["Dataset_IDs"][0],
            "Instrument": "SPY", "Candidate_ID": self.document["Allocation"]["Candidates"][0],
            "Decision_Time": "2026-11-10T21:01:00Z", "Execution_Time": "2026-11-11T14:30:00Z",
            "Inputs_Available_At": "2026-11-10T21:01:00Z", "Target_Weight": .5,
            "Artifacts": [{"Path": str(artifact), "SHA256": hashlib.sha256(artifact.read_bytes()).hexdigest()}]}

    def test_forward_commit_rejects_backfill_and_accepts_timely_signal(self):
        directory = self.freeze()
        verified = verify_freeze(directory, self.store)
        decision = self.decision(verified)
        spec = specification(self.document["Hypothesis"]["ID"])
        spec.update(Stage="forward", Configuration_SHA256=decision["Configuration_SHA256"], Source_Revision=decision["Source_Revision"],
                    Protocol_SHA256=decision["Protocol_SHA256"])
        with patch("regime_research.experiments.utc_now", return_value="2026-11-10T21:01:30+00:00"):
            register(self.store, "forward", spec)
            self.store.append("experiment_started", "forward", {}, idempotency_key="start")
        data = SimpleNamespace(bars=pd.DataFrame({"Available_At": pd.to_datetime([decision["Decision_Time"]]), "Instrument_ID": ["SPY"]}))
        with patch("regime_research.validation.load_dataset", return_value=data):
            with patch("regime_research.validation.utc_now", return_value="2026-11-11T14:30:00Z"):
                with self.assertRaisesRegex(ValueError, "before execution"):
                    record_forward_decision(self.project, directory, self.store, "forward", decision)
            with patch("regime_research.validation.utc_now", return_value="2026-11-10T21:01:10Z"):
                with self.assertRaisesRegex(ExperimentLogError, "aware and ordered"):
                    record_forward_decision(self.project, directory, self.store, "forward", decision)
            with patch("regime_research.validation.utc_now", return_value="2026-11-10T21:02:00Z"):
                event = record_forward_decision(self.project, directory, self.store, "forward", decision)
                self.assertEqual(event["Recorded_UTC"], "2026-11-10T21:02:00+00:00")
                self.assertEqual(record_forward_decision(self.project, directory, self.store, "forward", decision), event)
                altered = dict(decision, Target_Weight=.7)
                with self.assertRaisesRegex(ValueError, "cannot be replaced"):
                    record_forward_decision(self.project, directory, self.store, "forward", altered)

    def test_transfer_cannot_tune_recipe_or_select_an_unregistered_market(self):
        directory = self.freeze()
        verified = verify_freeze(directory, self.store)
        decision = self.decision(verified)
        spec = dict(specification(self.document["Hypothesis"]["ID"]), Stage="transfer",
            Configuration_SHA256=decision["Configuration_SHA256"], Source_Revision=decision["Source_Revision"],
            Protocol_SHA256=decision["Protocol_SHA256"])
        validate_run_binding(spec, verified, "QQQ")
        with self.assertRaises(ValueError):
            validate_run_binding(spec, verified, "unregistered")
        spec["Configuration_SHA256"] = "d"*64
        with self.assertRaises(ValueError):
            validate_run_binding(spec, verified, "QQQ")

    def test_historical_results_cannot_be_relabelled_as_forward(self):
        directory = self.freeze()
        verified = verify_freeze(directory, self.store)
        decisions = pd.DataFrame({"Decision_Time": ["2020-01-02T21:01Z"], "Execution_Time": ["2020-01-03T14:30Z"],
                                  "Return_End_Time": ["2020-01-06T14:30Z"]})
        self.assertFalse(validate_stage("retrospective", "SPY", decisions, verified, self.store)["Fresh_Time_Evidence"])
        with self.assertRaisesRegex(ValueError, "Historical observations"):
            validate_stage("forward", "SPY", decisions, verified, self.store)

    def test_registered_evaluation_archives_inputs_and_keeps_missing_audits_inconclusive(self):
        directory = self.freeze()
        verified = verify_freeze(directory, self.store)
        decision = self.decision(verified)
        spec = dict(specification(self.document["Hypothesis"]["ID"]), Stage="development",
            Configuration_SHA256=decision["Configuration_SHA256"], Source_Revision=decision["Source_Revision"],
            Protocol_SHA256=decision["Protocol_SHA256"])
        dates = pd.date_range("2020-01-06T14:30Z", periods=3, freq="B")
        decisions = pd.DataFrame({"Decision_Time": dates-pd.Timedelta(days=2), "Execution_Time": dates-pd.Timedelta(days=1),
                                  "Return_End_Time": dates, "Target_Weight": [1., 0., .5]})
        returns = pd.DataFrame({"Net_Return": [.01, -.01, .002], "Net_Excess_Return": [.01, -.01, .002]}, index=dates)
        with self.store.attempt(spec) as (identifier, outcome):
            result = record_evaluation(self.project, directory, self.store, identifier, "SPY", decisions,
                returns, returns, pd.Series([1., 0., .5], index=dates), returns, returns, validity={}, artifacts=[])
            outcome["Artifacts"] = result["Artifacts"]
        self.assertEqual(result["Stage"], "development")
        self.assertEqual(result["Gates"]["Promotion"], "withheld")
        self.assertEqual(result["Gates"]["Research_Validity_Status"], "inconclusive")
        self.assertEqual(len(result["Artifacts"]), 7)
        self.assertEqual(self.store.verify()["States"][identifier]["State"], "completed")
        with self.store.transaction():
            events, _, _ = self.store._read()
        event = next(value for value in events if value["Type"] == "evaluation_recorded")
        self.assertEqual(verify_evaluation(self.project, event, verified, self.store)["Observations"], 3)
        artifact = self.project/result["Artifacts"][0]["Path"]
        artifact.write_bytes(artifact.read_bytes()+b"tamper")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            verify_evaluation(self.project, event, verified, self.store)


class ChronologyTests(unittest.TestCase):
    def rows(self):
        base = pd.DataFrame({"Row_ID": [1], "Decision_Time": ["2020-01-02T21:01Z"],
                             "Available_At": ["2020-01-02T21:01Z"], "Label_End_Time": ["2020-01-03T14:30Z"]})
        cal = pd.DataFrame({"Row_ID": [2], "Decision_Time": ["2020-01-06T21:01Z"],
                            "Available_At": ["2020-01-06T21:01Z"], "Label_End_Time": ["2020-01-07T14:30Z"]})
        return base, cal

    def test_mature_purged_refit(self):
        self.assertEqual(validate_refit(*self.rows(), "2020-01-08T21:01Z")["Base_Rows"], 1)

    def test_base_and_calibration_label_boundaries_are_strict(self):
        for component in [0, 1]:
            base, cal = self.rows()
            if component == 0:
                base.loc[0, "Label_End_Time"] = cal.Decision_Time.iloc[0]
            else:
                cal.loc[0, "Label_End_Time"] = "2020-01-08T21:01Z"
            with self.assertRaisesRegex(ValueError, "Immature"):
                validate_refit(base, cal, "2020-01-08T21:01Z")

    def test_late_feature_and_duplicate_rows_fail(self):
        base, cal = self.rows()
        base.loc[0, "Available_At"] = "2020-01-02T21:02Z"
        with self.assertRaises(ValueError):
            validate_refit(base, cal, "2020-01-08T21:01Z")
        base, cal = self.rows()
        cal.loc[0, "Row_ID"] = 1
        with self.assertRaises(ValueError):
            validate_refit(base, cal, "2020-01-08T21:01Z")

    def test_selection_boundary_and_late_cash_are_rejected(self):
        returns = pd.DataFrame({"Return_End_Time": ["2020-01-02T21:01Z"]})
        validate_selection(returns, "2020-01-02T21:02Z")
        with self.assertRaises(ValueError):
            validate_selection(returns, "2020-01-02T21:01Z")
        with self.assertRaises(ValueError):
            validate_cash_timing(pd.DataFrame({"Available_At": ["2020-01-02T21:02Z"], "Interval_Start": ["2020-01-02T21:01Z"]}))

    def test_prefix_rebuild_catches_future_feature_leak(self):
        raw = pd.DataFrame({"Available_At": pd.to_datetime(["2020-01-01T21:01Z", "2020-01-02T21:01Z", "2020-01-03T21:01Z"]),
                            "Price": [1., 2., 5.]})
        def causal(frame, decision):
            return {"weight": float(frame.loc[frame.Available_At <= decision, "Price"].mean())}
        result = check_prefix_invariance(raw, causal, raw.Available_At.iloc[:2])
        self.assertEqual(len(result["Checked_Decisions"]), 2)
        def leaking(frame, decision):
            return {"weight": float(frame.Price.mean())}
        with self.assertRaisesRegex(ValueError, "Future data"):
            check_prefix_invariance(raw, leaking, raw.Available_At.iloc[:2])


class EvidenceTests(unittest.TestCase):
    def test_protocol_rejects_missing_commitments_and_benchmark_selection(self):
        document = protocol()
        del document["Models"]["Calibration"]
        with self.assertRaises(ValueError):
            validate_protocol(document)
        document = protocol()
        document["Allocation"]["Benchmark_Eligible"] = True
        with self.assertRaises(ValueError):
            validate_protocol(document)

    def test_negative_result_can_be_valid_research(self):
        metrics = {"Observations": 756, "Episodes": 10, "Delta_Sharpe": -.1, "Delta_Sharpe_Lower": -.2,
                   "Adjusted_P": .8, "DSR": .7, "Drawdown_Magnitude": .4, "Cost_Delta_Sharpe": -.1}
        gates = assess_gates(dict.fromkeys(VALIDITY_GATES, "passed"), metrics, protocol())
        self.assertEqual(gates["Research_Validity_Status"], "passed")
        self.assertEqual(gates["Strategy_Evidence_Status"], "failed")
        self.assertEqual(gates["Promotion"], "withheld")

    def test_insufficient_and_undefined_evidence_remain_inconclusive(self):
        gates = assess_gates({}, {"Observations": 755, "Episodes": 10, "Delta_Sharpe": 99}, protocol())
        self.assertEqual(set(gates["Strategy_Evidence"].values()), {"inconclusive"})
        metrics = {"Observations": 756, "Episodes": 10}
        gates = assess_gates(dict.fromkeys(VALIDITY_GATES, "passed"), metrics, protocol())
        self.assertEqual(gates["Strategy_Evidence_Status"], "inconclusive")

    def test_transfer_uses_all_registered_markets_fixed_weights_and_dates(self):
        frame = pd.DataFrame({"return": [.01, -.01]}, index=[1, 2])
        returns = {"QQQ": frame, "IEF": -frame}
        self.assertTrue(np.allclose(aggregate_transfer(returns, protocol()), 0))
        with self.assertRaises(ValueError):
            aggregate_transfer({"QQQ": frame}, protocol())
        with self.assertRaises(ValueError):
            aggregate_transfer({"QQQ": frame, "IEF": frame.iloc[1:]}, protocol())

    def test_evidence_computes_from_returns_and_rejects_post_horizon_extension(self):
        document = protocol()
        document["Inference"]["Thresholds"]["Minimum_Observations"] = 100
        document["Evaluation"]["Forward"]["Minimum_Completed_Intervals"] = 100
        document["Inference"]["Uncertainty"]["Replications"] = 49
        document["Inference"]["Multiplicity"]["Replications"] = 49
        generator = np.random.default_rng(42)
        baseline = generator.normal(.0002, .01, 100)
        candidate = baseline*.5+.0003
        frame = pd.DataFrame({"Net_Return": candidate, "Net_Excess_Return": candidate})
        control = pd.DataFrame({"Net_Return": baseline, "Net_Excess_Return": baseline})
        weights = pd.Series(np.tile([1., 0.], 50))
        metrics = evaluate_fixed_returns(frame, control, weights, frame, control, document)
        self.assertEqual(metrics["Episodes"], 50)
        self.assertTrue(np.isfinite(metrics["Delta_Sharpe"]))
        self.assertTrue(0 <= metrics["DSR"] <= 1)
        longer = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        with self.assertRaises(ValueError):
            evaluate_fixed_returns(longer, longer, pd.Series(1., index=longer.index), longer, longer, document)


if __name__ == "__main__":
    unittest.main()

"""Golden extraction, cache, interruption, receipt and publication regressions."""

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd

from regime_research.allocation import AllocationConfig, allocate, market_overlays
from regime_research.checkpoints import CheckpointStore
from regime_research.config import RunConfig
from regime_research.experiments import ExperimentStore, ZERO_HASH, content_hash, publish_directory
from regime_research.environment import verify_environment
from regime_research.features import build_features
from regime_research.labels import LabelConfig, build_labels
from regime_research.models import ModelConfig, fit_scores
from regime_research.research import analyze_returns
from regime_research.statistics import InferenceConfig
from regime_research.runner import execution_plan, run_notebook, verify_kernel_audit


PROJECT = Path(__file__).resolve().parents[2]


def synthetic_bars(rows=360, seed=1234):
    generator = np.random.default_rng(seed)
    close = 100*np.exp(np.cumsum(generator.normal(.0002, .01, rows)))
    bars = pd.DataFrame({"Open": close*.999, "High": close*1.01, "Low": close*.99, "Close": close,
                        "Volume": generator.integers(1000, 10000, rows)})
    dates = pd.bdate_range("2000-01-03", periods=rows)
    bars["Session"] = dates.strftime("%Y-%m-%d")
    bars["Open_Time"] = dates.tz_localize("UTC")+pd.Timedelta(hours=14, minutes=30)
    bars["Available_At"] = dates.tz_localize("UTC")+pd.Timedelta(hours=21, minutes=1)
    return bars


def recipe():
    return {"Recipe_ID": "test-downside", "Model": "HGB", "Target": "Binary_Downside", "Horizon": 10,
            "Training_Years": 5, "Feature_Group": "Full", "Refit_Frequency": "YS", "Complexity": 2}


class ExtractionTests(unittest.TestCase):
    def test_stale_local_package_install_is_rejected(self):
        original = metadata.version
        def stale(name):
            return "0.1.0" if name == "regime-research" else original(name)
        with patch("regime_research.environment.metadata.version", side_effect=stale):
            with self.assertRaisesRegex(ValueError, "local package"):
                verify_environment(PROJECT)
    def test_short_samples_produce_inconclusive_inference(self):
        frame = pd.DataFrame({"Net_Return": [.01, -.01], "Net_Excess_Return": [.01, -.01]})
        result = analyze_returns(frame, frame[["Net_Excess_Return"]], frame[["Net_Excess_Return"]], InferenceConfig(registered_trials=1))
        self.assertEqual(result.inference["Status"], "inconclusive_insufficient_observations")
        self.assertTrue(result.dsr_inputs.DSR.isna().all())
    def test_features_match_archived_notebook_golden_values(self):
        reference = json.loads((PROJECT/"tests/fixtures/features_reference.json").read_text())
        actual = build_features(synthetic_bars(reference["Rows"], reference["Seed"]))
        self.assertEqual(list(actual.columns), reference["Columns"])
        self.assertEqual(len(actual.columns), 98)
        np.testing.assert_allclose(actual.tail(10).to_numpy(), reference["Tail_Values"], rtol=1e-12, atol=1e-12)

    def test_feature_prefix_is_invariant_to_future_prices(self):
        bars = synthetic_bars()
        full, prefix = build_features(bars), build_features(bars.iloc[:300])
        pd.testing.assert_frame_equal(full.iloc[:300], prefix)

    def test_labels_match_next_open_formula_and_omit_exit_day_low(self):
        bars = synthetic_bars()
        labels = build_labels(bars, LabelConfig(horizon=5))
        self.assertAlmostEqual(labels.Forward_Return.iloc[50], bars.Open.iloc[56]/bars.Open.iloc[51]-1)
        self.assertEqual(labels.Label_End_Time.iloc[50], bars.Open_Time.iloc[56])
        changed = bars.copy()
        changed.loc[56, "Low"] = .01
        self.assertEqual(build_labels(changed, LabelConfig(horizon=5)).Downside_Risk.iloc[50], labels.Downside_Risk.iloc[50])
        self.assertTrue(labels.Binary_Downside.tail(6).isna().all())

    def test_allocation_preserves_prefix_and_rejects_lookahead_order(self):
        bars = synthetic_bars()
        scores = pd.DataFrame({"Row_ID": bars.index, "Risk_Score": np.linspace(0, 1, len(bars)),
                               "Decision_Time": bars.Available_At})
        config = AllocationConfig("test", "Always_Long", "Step", .75, .9, 10, 1260, .1, 5, None, 20)
        full = allocate(scores, market_overlays(bars), config)
        prefix = allocate(scores.iloc[:300], market_overlays(bars.iloc[:300]), config)
        pd.testing.assert_series_equal(full.Target_Weight.iloc[:300], prefix.Target_Weight)
        with self.assertRaises(ValueError):
            allocate(scores.iloc[::-1], market_overlays(bars), config)

    def test_configuration_rejects_unknown_fields_and_bad_dates(self):
        with self.assertRaises(ValueError):
            RunConfig(start="2026-01-01", end="2020-01-01")
        with self.assertRaises(ValueError):
            RunConfig(seed=True)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        base = PROJECT/"tmp/reproducibility_tests"
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = TemporaryDirectory(dir=base)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.bars = synthetic_bars(1600)
        self.features = pd.DataFrame({"Return": self.bars.Close.pct_change().fillna(0),
                                      "Range": self.bars.High/self.bars.Low-1})
        self.config = ModelConfig(minimum_base_rows=100, minimum_calibration_rows=30, selected_features=2)

    def scores(self, progress=None, features=None):
        return fit_scores(self.bars, self.features if features is None else features, recipe(), start="2005-01-03", end="2006-01-20",
            checkpoint_directory=self.directory, config=self.config, progress=progress)

    def test_cold_and_warm_cache_outputs_are_identical(self):
        cold = []
        first, first_audit = self.scores(cold.append)
        warm = []
        second, second_audit = self.scores(warm.append)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first_audit, second_audit)
        self.assertTrue(all(not row["Reused"] for row in cold))
        self.assertTrue(all(row["Reused"] for row in warm))

    def test_full_model_scores_survive_raw_prefix_truncation(self):
        full, _ = self.scores()
        decision = full.Decision_Time.iloc[min(260, len(full)-2)]
        prefix_bars = self.bars.loc[self.bars.Available_At <= decision]
        prefix_features = self.features.loc[prefix_bars.index]
        end = (decision.tz_convert("UTC").tz_localize(None).normalize()+pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        prefix, _ = fit_scores(prefix_bars, prefix_features, recipe(), start="2005-01-03", end=end,
            checkpoint_directory=self.directory/"prefix", config=self.config)
        expected = full.set_index("Row_ID").loc[prefix.Row_ID, "Risk_Score"].to_numpy()
        np.testing.assert_allclose(prefix.Risk_Score, expected, rtol=1e-12, atol=1e-12)

    def test_interrupt_preserves_finished_refits_for_resume(self):
        def interrupt(record):
            raise KeyboardInterrupt("simulated interruption after a completed refit")
        with self.assertRaises(KeyboardInterrupt):
            self.scores(interrupt)
        messages = []
        self.scores(messages.append)
        self.assertTrue(messages[0]["Reused"])
        self.assertTrue(any(not row["Reused"] for row in messages[1:]))

    def test_changed_inputs_invalidate_affected_checkpoints(self):
        self.scores()
        changed = self.features.copy()
        changed.loc[1000, "Return"] += .1
        messages = []
        self.scores(messages.append, changed)
        self.assertTrue(any(not row["Reused"] for row in messages))

    def test_corrupted_or_partial_checkpoint_is_rebuilt(self):
        first, _ = self.scores()
        checkpoint = next(self.directory.glob("*/scores.parquet"))
        checkpoint.write_bytes(b"interrupted")
        messages = []
        rebuilt, _ = self.scores(messages.append)
        pd.testing.assert_frame_equal(first, rebuilt)
        self.assertTrue(any(not row["Reused"] for row in messages))
        store = CheckpointStore(self.directory)
        self.assertIsNone(store.load({"uncommitted": True}))

    def test_runtime_and_scientific_sources_are_keyed_presentation_is_excluded(self):
        self.scores()
        receipt = json.loads(next(self.directory.glob("*/completion.json")).read_text())
        key = receipt["Key"]
        self.assertIn("scikit-learn", key["Runtime"])
        self.assertIn("features.py", key["Source_SHA256"])
        self.assertNotIn("reporting.py", key["Source_SHA256"])
        altered = dict(key, Runtime=dict(key["Runtime"], numpy="different"))
        self.assertIsNone(CheckpointStore(self.directory).load(altered))


class ReceiptTests(unittest.TestCase):
    def test_transient_directory_sharing_lock_is_retried(self):
        base = PROJECT/"tmp/reproducibility_tests"
        base.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=base) as temporary:
            root = Path(temporary)
            source, destination = root/"pending", root/"completed"
            source.mkdir()
            (source/"value").write_text("complete")
            original = os.rename
            calls = []
            def sharing_lock_then_rename(start, end):
                calls.append((start, end))
                if len(calls) == 1:
                    raise PermissionError("transient Windows sharing lock")
                return original(start, end)
            with patch("regime_research.experiments.os.rename", side_effect=sharing_lock_then_rename):
                publish_directory(source, destination, root=root)
            self.assertEqual((destination/"value").read_text(), "complete")
            self.assertEqual(len(calls), 2)
    def test_plan_rejects_duplicate_ids(self):
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("a=1"), nbformat.v4.new_code_cell("b=2")])
        notebook.cells[1].id = notebook.cells[0].id
        with self.assertRaises(ValueError):
            execution_plan(notebook)

    def test_extra_reordered_and_modified_execution_invalidate_receipts(self):
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("a=1")])
        plan = execution_plan(notebook)
        previous, events = ZERO_HASH, []
        for kind in ["cell_started", "cell_completed"]:
            event = {"Type": kind, "Previous_SHA256": previous, "Index": 0, "Cell_ID": plan[0]["Cell_ID"],
                     "Source_SHA256": plan[0]["Source_SHA256"], "Execution_Count": 1}
            event["Event_SHA256"] = content_hash(event)
            previous = event["Event_SHA256"]
            events.append(event)
        verify_kernel_audit(plan, events)
        for altered in [events+events, events[::-1], [dict(events[0], Cell_ID="extra"), events[1]]]:
            with self.assertRaises(ValueError):
                verify_kernel_audit(plan, altered)

    def test_failed_supported_run_has_no_completion_and_records_failure(self):
        base = PROJECT/"tmp/reproducibility_tests"
        base.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=base) as temporary:
            project = Path(temporary)
            (project/"protocols").mkdir()
            (project/"protocols/layer6_validation_v2.json").write_bytes((PROJECT/"protocols/layer6_validation_v2.json").read_bytes())
            (project/"requirements.lock").write_text("example==1\n")
            source = {"module.py": "a"*64}
            with patch("regime_research.runner.verify_environment", return_value={"Packages": {}}), \
                 patch("regime_research.runner.scientific_sources", return_value=source), \
                 patch("regime_research.runner.git_revision", return_value={"Commit": "test"}), \
                 patch("regime_research.runner.execute_cells", side_effect=TimeoutError("simulated interrupted kernel")):
                with self.assertRaises(TimeoutError):
                    run_notebook(project, project/"run", RunConfig())
            self.assertFalse((project/"run/completion.json").exists())
            self.assertTrue((project/"run/failure.json").exists())
            states = ExperimentStore(project/"experiments").verify()["States"]
            self.assertEqual(next(iter(states.values()))["State"], "failed")


if __name__ == "__main__":
    unittest.main()

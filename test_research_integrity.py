"""Regression tests for the six research-review corrections."""

import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from research_integrity import (
    downside_deviation,
    equity_drawdown,
    executed_notebook_digest,
    file_digest,
    forward_adverse_return,
    frame_digest,
    load_predictor_cache,
    realized_selection_rows,
    source_provenance,
    verify_run_provenance,
    write_run_manifest,
)


PROJECT_DIR = Path(__file__).resolve().parent


def load_functions(source, names, namespace):
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<regression>", "exec"), namespace)


class MetricAndTimingTests(unittest.TestCase):
    def test_downside_deviation_is_rms_shortfall(self):
        self.assertAlmostEqual(downside_deviation([-.1, .2, 0]), np.sqrt(.01 / 3))
        self.assertEqual(downside_deviation([.1, .2]), 0)

    def test_initial_capital_is_included_in_drawdown(self):
        np.testing.assert_allclose(equity_drawdown(pd.Series([.9, .9])), [-.1, -.1])
        np.testing.assert_allclose(equity_drawdown(pd.Series([1.1, .99])), [0, -.1])

    def test_adverse_target_excludes_exit_day_low_but_includes_exit_open(self):
        opens = pd.Series([100., 100., 100., 90., 100.])
        lows = pd.Series([100., 98., 97., 1., 99.])
        self.assertAlmostEqual(forward_adverse_return(opens, lows, 2).iloc[0], -.1)
        opens.iloc[3] = 100.
        self.assertAlmostEqual(forward_adverse_return(opens, lows, 2).iloc[0], -.03)
        self.assertTrue(forward_adverse_return(opens, lows, 2).tail(3).isna().all())

    def test_selection_excludes_unfinished_pnl(self):
        frame = pd.DataFrame({
            "Date": pd.to_datetime(["2007-01-02", "2007-01-03", "2007-01-04"]),
            "Return_End_Date": pd.to_datetime(["2007-01-04", "2007-01-05", "2007-01-08"]),
        })
        selected = realized_selection_rows(frame, pd.Timestamp("2005-01-05"),
                                           pd.Timestamp("2007-01-05"), pd.Timestamp("2007-01-05"))
        self.assertEqual(selected.index.tolist(), [0])

    def test_nested_selector_ignores_future_return_perturbation(self):
        source = (PROJECT_DIR / "sharpe_improvement_research.py").read_text()
        namespace = dict(np=np, pd=pd, ANNUALIZATION_DAYS=252,
                         SHARPE_OUTER_YEARS=2, SHARPE_INNER_FOLDS=4, SHARPE_INNER_YEARS=2,
                         SHARPE_TURNOVER_PENALTY=.02, SHARPE_RISK_FREE_ANNUAL=0.,
                         SHARPE_OUTER_START=pd.Timestamp("2007-01-05"),
                         SHARPE_OUTER_END=pd.Timestamp("2009-01-05"),
                         ADVANCED_ONE_WAY_COST=.0005, downside_deviation=downside_deviation,
                         equity_drawdown=equity_drawdown, realized_selection_rows=realized_selection_rows)
        load_functions(source, {"sharpe_metric_record", "sharpe_recompute_returns",
                                "sharpe_intervals", "sharpe_select_nested"}, namespace)
        dates = pd.bdate_range("1999-01-05", "2009-01-08")
        frame = pd.DataFrame({"Date": dates, "Execution_Date": pd.Series(dates).shift(-1),
                              "Return_End_Date": pd.Series(dates).shift(-2),
                              "Tradable_Return_1d": np.sin(np.arange(len(dates))) * .01 + .0005,
                              "Execution_Volatility": .15, "Position": 1.})
        base = namespace["sharpe_recompute_returns"](frame, .0005)
        namespace["sharpe_candidate_backtests"] = {"A": base.copy(), "B": base.copy()}
        namespace["sharpe_candidate_registry"] = pd.DataFrame({"Candidate_ID": ["A", "B"], "Complexity": [0, 1]})
        inner, before, _ = namespace["sharpe_select_nested"]()
        self.assertTrue(inner.Return_Data_End.lt(inner.Outer_Start).all())
        unavailable = (base.Date < pd.Timestamp("2007-01-05")) & (base.Return_End_Date >= pd.Timestamp("2007-01-05"))
        namespace["sharpe_candidate_backtests"]["B"].loc[unavailable, "Net_Excess_Return"] = 1000.
        _, after, _ = namespace["sharpe_select_nested"]()
        self.assertEqual(before.Selected_Candidate.tolist(), after.Selected_Candidate.tolist())
        self.assertEqual(before.Median_Inner_Sharpe.tolist(), after.Median_Inner_Sharpe.tolist())

    def test_all_metric_entry_points_use_corrected_calculations(self):
        notebook = json.loads((PROJECT_DIR / "sp500_market_regime_classification.ipynb").read_text())
        sources = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
        references = json.loads((PROJECT_DIR / "tests/fixtures/legacy_metric_functions.json").read_text())
        for identifier, source in references.items():
            sources.setdefault(identifier, source)
        namespace = dict(np=np, pd=pd, ANNUALIZATION_DAYS=252, TRADING_DAYS_PER_YEAR=252,
                         equity_drawdown=equity_drawdown, downside_deviation=downside_deviation,
                         rms_downside_deviation=downside_deviation)
        load_functions(sources["regime-strategy-backtest"], {"performance_statistics"}, namespace)
        load_functions(sources["advanced-strategy-helpers"], {"advanced_performance_metrics"}, namespace)
        load_functions((PROJECT_DIR / "sharpe_improvement_research.py").read_text(), {"sharpe_metric_record"}, namespace)
        frame = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=3),
                              "Net_Return": [-.1, .02, 0.], "Net_Excess_Return": [-.1, .02, 0.],
                              "Gross_Return": [-.1, .02, 0.], "Position": [1., 1., 1.],
                              "Turnover": [1., 0., 0.], "Transaction_Cost": [0., 0., 0.]})
        frame["Net_Equity"] = (1 + frame.Net_Return).cumprod()
        frame["Gross_Equity"] = frame.Net_Equity
        frame["Drawdown"] = equity_drawdown(frame.Net_Equity)
        expected_sortino = frame.Net_Return.mean() / np.sqrt(.01 / 3) * np.sqrt(252)
        for name in ["advanced_performance_metrics", "sharpe_metric_record"]:
            metrics = namespace[name](frame)
            self.assertAlmostEqual(metrics["Sortino_Ratio"], expected_sortino)
            self.assertAlmostEqual(metrics["Maximum_Drawdown"], -.1)
        self.assertAlmostEqual(namespace["performance_statistics"](frame.Net_Return)["Maximum_Drawdown"], -.1)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT_DIR)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scores = self.root / "scores.csv"
        self.audit = self.root / "audit.csv"
        self.manifest = self.root / "manifest.json"
        self.dates = pd.date_range("2020-01-01", periods=2)
        self.score_frame = pd.DataFrame({
            "Recipe_ID": ["A", "A"], "Date": self.dates, "Risk_Score": [.2, .3],
            "Execution_Date": self.dates + pd.Timedelta(days=1),
            "Return_End_Date": self.dates + pd.Timedelta(days=2),
            "Label_End_Date": self.dates + pd.Timedelta(days=5),
            "Refit_Date": [self.dates[0]] * 2,
        })
        self.audit_frame = pd.DataFrame({
            "Recipe_ID": ["A"], "Refit_Date": [self.dates[0]],
            "Prediction_End": [self.dates[-1]], "Training_Start": ["2010-01-01"],
            "Training_End": ["2018-12-01"], "Training_Label_End": ["2018-12-20"],
            "Calibration_Start": ["2019-01-01"], "Calibration_Label_End": ["2019-12-30"],
            "Calibration_Rows": [126],
        })
        self.publish()

    def publish(self):
        self.score_frame.to_csv(self.scores, index=False)
        self.audit_frame.to_csv(self.audit, index=False)
        self.manifest.write_text(json.dumps({
            "Predictor_Cache_Signature": "current", "Score_Rows": len(self.score_frame),
            "Audit_Rows": len(self.audit_frame), "Score_SHA256": file_digest(self.scores),
            "Audit_SHA256": file_digest(self.audit),
        }))

    def load(self, signature="current"):
        return load_predictor_cache(self.scores, self.audit, self.manifest,
                                    signature, {"A"}, self.dates)

    def test_matching_cache_loads(self):
        self.assertIsNotNone(self.load())

    def test_missing_or_stale_manifest_rejects(self):
        self.assertIsNone(self.load("different"))
        self.manifest.unlink()
        self.assertIsNone(self.load())

    def test_damaged_cache_rejects(self):
        self.scores.write_text(self.scores.read_text().replace("0.2", "0.9"))
        self.assertIsNone(self.load())

    def test_incomplete_dates_or_unpurged_training_rejects(self):
        self.score_frame = self.score_frame.iloc[:1]
        self.publish()
        self.assertIsNone(self.load())
        self.audit_frame.loc[0, "Training_Label_End"] = "2020-01-01"
        self.publish()
        self.assertIsNone(self.load())

    def test_high_volume_and_feature_revisions_change_fingerprint(self):
        frame = pd.DataFrame({"Open": [100.], "High": [102.], "Low": [98.],
                              "Close": [101.], "Volume": [1000], "Feature": [.5]})
        for column in ["High", "Volume", "Feature"]:
            changed = frame.copy()
            changed[column] += 1
            self.assertNotEqual(frame_digest(frame), frame_digest(changed))


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT_DIR)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "model_outputs").mkdir()
        for name in ["sp500_ohlcv.csv", "sharpe_improvement_research.py", "research_integrity.py"]:
            (self.root / name).write_text("original")
        self.path = self.root / "sp500_market_regime_classification.ipynb"
        self.notebook = {"cells": [
            {"id": "first", "cell_type": "code", "source": ["x = 1\n"]},
            {"id": "sharpe-experiments", "cell_type": "code", "source": ["run()\n"], "outputs": []},
        ]}
        self.save_notebook()
        (self.root / "model_outputs/35_daily.csv").write_text("return\n0.1\n")
        self.digest = executed_notebook_digest(self.path, ["", "x = 1", "run()"], "sharpe-experiments")
        run_id = write_run_manifest(self.root, source_provenance(self.root), self.digest)
        self.notebook["cells"][1]["outputs"] = [{"text": [f"Research run completed: {run_id}\n"]}]
        self.save_notebook()

    def save_notebook(self):
        self.path.write_text(json.dumps(self.notebook))

    def test_complete_current_run_verifies(self):
        verify_run_provenance(self.root)

    def test_unexecuted_code_edit_rejects_old_outputs(self):
        self.notebook["cells"][1]["source"] = ["raise RuntimeError('unexecuted')\n"]
        self.save_notebook()
        with self.assertRaisesRegex(AssertionError, "source or data"):
            verify_run_provenance(self.root)

    def test_changed_data_or_model_source_rejects(self):
        for name in ["sp500_ohlcv.csv", "sharpe_improvement_research.py"]:
            path = self.root / name
            path.write_text("changed")
            with self.assertRaises(AssertionError):
                verify_run_provenance(self.root)
            path.write_text("original")

    def test_missing_completion_receipt_rejects(self):
        self.notebook["cells"][1]["outputs"] = []
        self.save_notebook()
        with self.assertRaisesRegex(AssertionError, "receipt"):
            verify_run_provenance(self.root)

    def test_modified_artifact_rejects(self):
        (self.root / "model_outputs/35_daily.csv").write_text("return\n9.9\n")
        with self.assertRaisesRegex(AssertionError, "checksum"):
            verify_run_provenance(self.root)

    def test_wrong_kernel_history_rejects(self):
        with self.assertRaises(RuntimeError):
            executed_notebook_digest(self.path, ["", "x = 2", "run()"], "sharpe-experiments")


if __name__ == "__main__":
    unittest.main()

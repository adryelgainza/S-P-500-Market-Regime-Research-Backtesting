"""Regressions for control chronology, common accounting, and attribution."""

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from regime_research.attribution import _factor_regression, _holm, attribute, predictive_diagnostics
from regime_research.controls import build_controls, shuffle_overlay_blocks, shuffled_score_null
from regime_research.execution import ExecutionConfig, simulate_portfolio
from tests.unit.test_execution import inputs


def protocol():
    data = json.loads((Path(__file__).resolve().parents[2]/"protocols/layer4_controls.json").read_text())
    data.update(One_Way_Cost=0., HAC_Lag=1, Bootstrap_Replications=49,
                Bootstrap_Block_Length=2, Reference_Portfolio="Model", Primary_Comparisons=["Control"])
    return data


def environment(result, cash_rate=0.):
    states = result.valuations
    returns = states.Price.to_numpy()[1:]/states.Price.to_numpy()[:-1]-1
    return pd.DataFrame(dict(Market_Return=returns, Market_Excess_Return=returns-cash_rate,
        Reference_Return=cash_rate, Credited_Return=cash_rate, Fold=1),
        index=pd.DatetimeIndex(result.daily.Return_End_Time))


def control_inputs():
    evaluation = pd.DataFrame(dict(Decision_Time=pd.to_datetime(["2020-01-02T21:01Z", "2020-01-03T21:01Z"]),
        Execution_Time=pd.to_datetime(["2020-01-03T14:30Z", "2020-01-06T14:30Z"]),
        Return_End_Time=pd.to_datetime(["2020-01-06T14:30Z", "2020-01-07T14:30Z"]),
        Fold=1, SMA_Signal=[1.,0], Volatility=.2, Core_Weight=1.))
    history = pd.DataFrame(dict(Fold=1, Candidate_ID="chosen",
        Decision_Time=pd.to_datetime(["2019-11-01T21:01Z", "2019-11-04T21:01Z"]),
        Return_End_Time=pd.to_datetime(["2019-11-05T14:30Z", "2019-11-06T14:30Z"]),
        Target_Weight=[.2,.6], Net_Excess_Return=[-.01,.01], Market_Excess_Return=[-.02,.02]))
    folds = pd.DataFrame(dict(Fold=[1], Selected_Candidate=["chosen"], Decision_Time=pd.to_datetime(["2020-01-01T00:00Z"])))
    return dict(evaluation=evaluation, history=history, folds=folds)


class ControlTests(unittest.TestCase):
    def test_matching_ignores_unrealized_or_future_returns(self):
        settings = protocol()
        settings["Minimum_Matching_Rows"] = 2
        data = control_inputs()
        original = build_controls(settings, data)
        late = data["history"].iloc[[0]].copy()
        late["Return_End_Time"] = pd.Timestamp("2020-01-03T14:30Z")
        late["Net_Excess_Return"] = 100000.
        late["Target_Weight"] = 1.
        data["history"] = pd.concat([data["history"], late], ignore_index=True)
        changed = build_controls(settings, data)
        pd.testing.assert_frame_equal(original.weights, changed.weights)
        self.assertAlmostEqual(original.weights.Past_Exposure_Match.iloc[0], .4)
        self.assertAlmostEqual(original.weights.Past_Volatility_Match.iloc[0], .5)
        self.assertTrue(original.matching_audit.Estimation_End.lt(original.matching_audit.Matching_Decision_Time).all())

    def test_matching_rejects_wrong_candidate_and_late_fold_decisions(self):
        settings = protocol()
        settings["Minimum_Matching_Rows"] = 2
        data = control_inputs()
        data["history"]["Candidate_ID"] = "different"
        with self.assertRaisesRegex(ValueError, "selected candidate"):
            build_controls(settings, data)
        data = control_inputs()
        data["folds"]["Decision_Time"] = pd.Timestamp("2020-01-03T22:00Z")
        with self.assertRaisesRegex(ValueError, "after evaluation"):
            build_controls(settings, data)

    def test_randomization_preserves_whole_blocks_and_fold_marginals(self):
        values = np.arange(24)/24
        folds = np.repeat([1,2], 12)
        result = shuffle_overlay_blocks(values, folds, block_length=3, seed=913)
        np.testing.assert_array_equal(result, shuffle_overlay_blocks(values, folds, block_length=3, seed=913))
        for fold in [1,2]:
            original, shuffled = values[folds==fold], result[folds==fold]
            np.testing.assert_array_equal(np.sort(original), np.sort(shuffled))
            self.assertEqual({tuple(original[i:i+3]) for i in range(0,12,3)},
                             {tuple(shuffled[i:i+3]) for i in range(0,12,3)})

    def test_shuffled_score_null_uses_predeclared_seed_and_gate(self):
        settings = protocol()
        draws, summary = shuffled_score_null(settings)
        self.assertEqual(len(draws), 400)
        self.assertTrue(summary["Passed"], summary)


class AttributionTests(unittest.TestCase):
    def test_saved_factor_rows_are_compared_by_portfolio_identifier(self):
        from regime_research.attribution_research import _compare_attribution_table
        expected = pd.DataFrame({"Status":["ok","ok"], "Daily_Alpha":[.01,.02], "Portfolio":["A","B"]})
        _compare_attribution_table(expected, expected.iloc[::-1])
        changed = expected.copy()
        changed.loc[0,"Daily_Alpha"] = .03
        with self.assertRaises(AssertionError):
            _compare_attribution_table(expected, changed)

    def test_components_and_fold_episode_logs_reconcile(self):
        args = inputs([100,100,110,95,105,100], [.5]*4)
        model = simulate_portfolio(*args, ExecutionConfig(one_way_cost=0))
        other = simulate_portfolio(*inputs([100,100,110,95,105,100], [1.]*4), ExecutionConfig(one_way_cost=0))
        result = attribute({"Model": model, "Control": other}, environment(model), protocol())
        self.assertTrue(result.attribution.Pnl_Bridge_Error.abs().lt(1e-12).all())
        for name, portfolio in [("Model",model),("Control",other)]:
            expected = np.log(portfolio.daily.Net_Equity.iloc[-1])
            self.assertAlmostEqual(result.episodes.loc[result.episodes.Portfolio.eq(name), "Log_Contribution"].sum(), expected)
            self.assertAlmostEqual(result.fold_contributions.loc[result.fold_contributions.Portfolio.eq(name), "Log_Contribution"].sum(), expected)

    def test_exit_cost_belongs_to_the_invested_episode(self):
        args = inputs([100]*5, [1,0,0])
        model = simulate_portfolio(*args, ExecutionConfig(one_way_cost=.01))
        settings = protocol()
        settings["One_Way_Cost"] = .01
        result = attribute({"Model": model, "Control": model}, environment(model), settings)
        episode = result.episodes.loc[result.episodes.Portfolio.eq("Model") & result.episodes.Episode_ID.eq(1)].iloc[0]
        self.assertEqual(episode.Holding_Periods, 1)
        self.assertEqual(episode.Accounting_Periods, 2)
        self.assertAlmostEqual(episode.Trading_Cost, model.trades.Cost.sum())

    def test_different_cash_crediting_is_detected(self):
        args = list(inputs([100,100,110,95,105,100], [.5]*4, cash_rate=.001))
        model = simulate_portfolio(*args, ExecutionConfig(one_way_cost=0))
        args[2] = args[2].copy()
        args[2]["Credited_Return"] = .002
        other = simulate_portfolio(*args, ExecutionConfig(one_way_cost=0))
        with self.assertRaisesRegex(ValueError, "credited cash"):
            attribute({"Model": model, "Control": other}, environment(model, .001), protocol())

    def test_different_costs_are_detected(self):
        args = inputs([100,100,110,95,105,100], [.5]*4)
        model = simulate_portfolio(*args, ExecutionConfig(one_way_cost=0))
        other = simulate_portfolio(*args, ExecutionConfig(one_way_cost=.01))
        with self.assertRaisesRegex(ValueError, "assumptions differ"):
            attribute({"Model": model, "Control": other}, environment(model), protocol())

    def test_constant_beta_has_exact_zero_alpha(self):
        market = np.random.default_rng(13).normal(0,.01,1000)
        result = _factor_regression(.6*market, market, 20, .95)
        self.assertAlmostEqual(result["Beta"], .6)
        self.assertAlmostEqual(result["Daily_Alpha"], 0, places=12)

    def test_holm_retains_the_declared_comparison_family(self):
        np.testing.assert_allclose(_holm([.04,.01,.03]), [.06,.03,.06])

    def test_predictive_diagnostics_exclude_unrealized_labels(self):
        frame = pd.DataFrame(dict(Recipe_ID="risk",
            Decision_Time=pd.to_datetime(["2020-01-01T21:00Z", "2020-01-02T21:00Z", "2020-01-03T21:00Z"]),
            Label_End_Time=pd.to_datetime(["2020-01-02T14:30Z", "2020-01-03T14:30Z", "2020-02-01T14:30Z"]),
            Actual_Event=[0,1,1], Risk_Score=[.1,.9,.5], Predicted_Probability=[.1,.9,.5],
            Forward_Return=[.01,-.01,-10.]))
        summary, calibration, quantiles = predictive_diagnostics(frame, protocol(), evaluation_end_time="2020-01-04T00:00Z")
        self.assertEqual(summary.Rows.iloc[0], 2)
        self.assertEqual(summary.ROC_AUC.iloc[0], 1)
        self.assertAlmostEqual(summary.Brier.iloc[0], .01)
        self.assertEqual(calibration.Rows.sum(), 2)
        self.assertEqual(quantiles.Rows.sum(), 2)


if __name__ == "__main__":
    unittest.main()

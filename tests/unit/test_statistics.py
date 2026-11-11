"""Regressions for published DSR inputs, paired resampling, and HAC inference."""

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from regime_research.statistics import (InferenceConfig, annualized_sharpe, deflated_sharpe,
    deflated_sharpe_from_moments, hac_long_run_covariance, hac_market_regression, hac_mean_test,
    infer_performance, max_t_adjustment, paired_sharpe_bootstrap, trial_family_diagnostics)


class DeflatedSharpeTests(unittest.TestCase):
    def test_published_example_and_annual_to_daily_variance_conversion(self):
        fixture = json.loads((Path(__file__).resolve().parents[1]/"fixtures/dsr_published_example.json").read_text())
        for case in fixture["Cases"]:
            row = deflated_sharpe_from_moments(fixture["Annual_Sharpe"]/np.sqrt(fixture["Annualization"]),
                fixture["Observations"], case["Skewness"], case["Nonexcess_Kurtosis"],
                fixture["Annual_Trial_Sharpe_Variance"]/fixture["Annualization"], case["Independent_Trials"])
            self.assertEqual(row["Status"], "ok")
            self.assertEqual(round(row["DSR"], 4), case["Expected_DSR_Rounded"])
            if "Expected_Threshold_Rounded" in case:
                self.assertEqual(round(row["Expected_Max_Daily_Sharpe"], 4), case["Expected_Threshold_Rounded"])

    def test_trial_dispersion_comes_from_candidate_sharpes(self):
        rng = np.random.default_rng(7)
        family = pd.DataFrame(rng.normal(size=(600, 4)) + np.arange(4)*.03)
        rows, summary, _, _ = deflated_sharpe(family[0].to_numpy(), family, 8)
        expected = np.var(family.mean()/family.std(ddof=1), ddof=1)
        self.assertAlmostEqual(summary["Trial_Sharpe_Variance"], expected)
        self.assertEqual(summary["Unavailable_Registered_Trials"], 4)
        self.assertEqual(rows.Independent_Trials.iloc[0], 8)

    def test_duplicates_zero_dispersion_and_single_independent_trial(self):
        values = np.random.default_rng(8).normal(.02, 1, 600)
        family = pd.DataFrame({"first": values, "duplicate": values})
        rows, summary, diagnostics, _ = deflated_sharpe(values, family, 2)
        self.assertEqual(summary["Distinct_Usable_Return_Streams"], 1)
        self.assertEqual(summary["Rounded_Independent_Trials"], 1)
        self.assertEqual(summary["Trial_Sharpe_Variance"], 0)
        self.assertEqual(diagnostics.Duplicate_Of.iloc[1], "first")
        self.assertAlmostEqual(rows.DSR.iloc[0], rows.DSR.iloc[1])
        single = deflated_sharpe_from_moments(.05, 600, 0, 3, None, 1)
        self.assertEqual(single["Status"], "ok")
        self.assertEqual(single["Expected_Max_Daily_Sharpe"], 0)

    def test_negative_correlations_are_explicitly_unsupported(self):
        values = np.random.default_rng(9).normal(size=600)
        family = pd.DataFrame({"positive": values, "negative": -values})
        rows, summary, _, _ = deflated_sharpe(values, family, 2)
        self.assertIsNone(summary["Rounded_Independent_Trials"])
        self.assertIn("unsupported", rows.Status.iloc[1])
        self.assertEqual(rows.Status.iloc[0], "ok")

    def test_unavailable_and_constant_trials_remain_disclosed(self):
        values = np.random.default_rng(3).normal(size=300)
        family = pd.DataFrame({"valid": values, "cash": 0., "missing": np.nan})
        summary, table, _ = trial_family_diagnostics(family, 5)
        self.assertEqual(summary["Usable_Trials"], 1)
        self.assertEqual(summary["Unavailable_Registered_Trials"], 4)
        self.assertEqual(table.Status.tolist(), ["ok", "zero_return_variance", "missing_or_nonfinite_returns"])
        rows, _, _, _ = deflated_sharpe(np.zeros(300), family, 5)
        self.assertTrue(rows.Status.eq("undefined_selected_series").all())

    def test_invalid_variance_expression_is_undefined(self):
        row = deflated_sharpe_from_moments(1, 100, 5, 1, .01, 10)
        self.assertEqual(row["Status"], "nonpositive_sharpe_variance")
        self.assertIsNone(row["DSR"])


class DependenceTests(unittest.TestCase):
    def test_hac_matches_manual_bartlett_mean_variance(self):
        values = np.array([1., -1, 2, .2, 1, -.5, .8, 2])
        centered = values-values.mean()
        variance = np.dot(centered, centered)/len(values)
        variance += 2*(1-1/3)*np.dot(centered[1:], centered[:-1])/len(values)
        variance += 2*(1-2/3)*np.dot(centered[2:], centered[:-2])/len(values)
        result = hac_mean_test(values, 2)
        self.assertAlmostEqual(result["HAC_SE"], np.sqrt(variance/len(values)))
        self.assertAlmostEqual(hac_long_run_covariance(values, 2)[0,0], variance)

    def test_positive_autocorrelation_increases_hac_uncertainty(self):
        rng = np.random.default_rng(62)
        values = np.zeros(3000)
        for index in range(1, len(values)):
            values[index] = .7*values[index-1]+rng.normal()
        self.assertGreater(hac_mean_test(values, 20)["HAC_SE"], 1.5*hac_mean_test(values, 0)["HAC_SE"])

    def test_market_regression_recovers_planted_alpha_and_beta(self):
        rng = np.random.default_rng(2)
        market = rng.normal(0, .01, 3000)
        noise = rng.normal(0, .002, 3000)
        noise -= noise.mean()
        centered_market = market-market.mean()
        noise -= centered_market*np.dot(noise, centered_market)/np.dot(centered_market, centered_market)
        strategy = .0003+.8*market+noise
        result = hac_market_regression(strategy, market, 20)
        self.assertAlmostEqual(result["Beta"], .8, delta=.02)
        self.assertLess(result["Alpha_Lower"], .0003)
        self.assertGreater(result["Alpha_Upper"], .0003)
        self.assertEqual(hac_market_regression(strategy, np.zeros(3000))["Status"], "rank_deficient_market")

    def test_paired_bootstrap_preserves_identical_series_and_seed(self):
        values = np.random.default_rng(4).normal(0, .01, 211)
        first = paired_sharpe_bootstrap(values, values, repetitions=201, block_length=20, seed=91)
        second = paired_sharpe_bootstrap(values, values, repetitions=201, block_length=20, seed=91)
        self.assertEqual(first, second)
        self.assertEqual(first["Delta_Sharpe_Lower"], 0)
        self.assertEqual(first["Delta_Sharpe_Upper"], 0)
        bad = paired_sharpe_bootstrap(values, np.zeros(211), repetitions=30)
        self.assertIn("insufficient", bad["Status"])

    def test_max_t_preserves_duplicate_trials_and_adjusts_jointly(self):
        values = np.random.default_rng(15).normal(.001, .01, 311)
        family = pd.DataFrame({"first": values, "duplicate": values, "zero": 0.})
        result = max_t_adjustment(family, np.zeros(311), repetitions=199, seed=7)
        self.assertEqual(result.MaxT_Adjusted_P.iloc[0], result.MaxT_Adjusted_P.iloc[1])
        self.assertGreaterEqual(result.MaxT_Adjusted_P.iloc[0], result.Unadjusted_Bootstrap_P.iloc[0])
        self.assertEqual(result.Status.iloc[2], "zero_hac_variance")
        self.assertTrue(np.isnan(result.MaxT_Adjusted_P.iloc[2]))
        np.testing.assert_allclose(result.HAC_SE.iloc[0], hac_mean_test(values, 20)["HAC_SE"])


class InferenceContractTests(unittest.TestCase):
    def test_csv_verification_normalizes_missing_diagnostic_fields(self):
        from regime_research.statistical_research import _compare_saved_table
        saved = pd.DataFrame({"Duplicate_Of": [np.nan], "DSR": [.5]})
        computed = pd.DataFrame({"Duplicate_Of": [None], "DSR": [.5+1e-14]})
        _compare_saved_table(saved, computed)
        with self.assertRaises(AssertionError):
            _compare_saved_table(saved, pd.DataFrame({"Duplicate_Of": ["changed"], "DSR": [.5]}))

    def test_zero_variance_results_have_explicit_statuses_and_table_schemas(self):
        index = pd.date_range("2020-01-01", periods=250, tz="UTC")
        values = np.random.default_rng(22).normal(0, .01, 250)
        portfolio = pd.DataFrame({"Net_Return": 0., "Net_Excess_Return": 0.}, index=index)
        benchmarks = pd.DataFrame({"market": values}, index=index)
        trials = pd.DataFrame({"all_cash": 0.}, index=index)
        config = InferenceConfig(registered_trials=1, primary_bootstrap_reps=19,
                                  sensitivity_bootstrap_reps=19, max_t_reps=19)
        result = infer_performance(portfolio, benchmarks, trials, config)
        self.assertIsNone(result.inference["Sharpe"])
        self.assertIsNone(result.inference["Sortino"])
        self.assertTrue(result.dsr_inputs.Status.eq("undefined_selected_series").all())
        self.assertEqual(result.trial_dependence.columns.tolist(), ["Candidate_ID", "Mean_Correlation_With_Other_Trials"])

    def test_aligned_interface_and_undefined_sortino(self):
        rng = np.random.default_rng(12)
        index = pd.date_range("2020-01-01", periods=250, tz="UTC")
        values = rng.normal(.0001, .01, (250, 4))
        portfolio = pd.DataFrame({"Net_Return": values[:, 0], "Net_Excess_Return": values[:, 0]}, index=index)
        benchmarks = pd.DataFrame({"HGB": values[:,1], "Market": values[:,2]}, index=index)
        trials = pd.DataFrame(values, index=index)
        config = InferenceConfig(registered_trials=4, primary_bootstrap_reps=49,
                                  sensitivity_bootstrap_reps=29, max_t_reps=49)
        result = infer_performance(portfolio, benchmarks, trials, config)
        self.assertEqual(len(result.bootstrap_summary), 6)
        self.assertEqual(len(result.hac_summary), 6)
        self.assertEqual(len(result.max_t), 5)
        self.assertEqual(result.inference["Selector_Inference_Scope"], "conditional_saved_selector_stream")
        with self.assertRaises(ValueError):
            infer_performance(portfolio, benchmarks.iloc[::-1], trials, config)
        with self.assertRaises(ValueError):
            InferenceConfig(registered_trials=0)

    def test_daily_sharpe_annualization(self):
        values = np.array([-.01, .02, .03, -.005])
        self.assertAlmostEqual(float(annualized_sharpe(values, 250)), values.mean()/values.std(ddof=1)*np.sqrt(250))


if __name__ == "__main__":
    unittest.main()

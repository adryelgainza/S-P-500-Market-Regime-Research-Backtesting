import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from regime_research.data import (
    CashPolicy, DataValidationError, HistoryRequirements, ValidationPolicy,
    accrue_quoted_rate, build_cash_intervals, build_eligible_sample, empty_table,
    exchange_schedule, load_dataset, normalize_bars, restore_as_traded_history,
    write_snapshot,
)
from regime_research.data.adapters import parse_fred_dff, yahoo_history_tables


def market_fixture(start="2024-01-02", end="2024-01-09"):
    schedule = exchange_schedule(start, end)
    raw = pd.DataFrame({"Session": schedule.Session, "Open": 100., "High": 101.,
                        "Low": 99., "Close": 100., "Volume": 1000.})
    return normalize_bars(raw, schedule, instrument="SPY", source="synthetic"), schedule


def metadata_fixture():
    return dict(Track="synthetic", Instrument_ID="SPY", Calendar="NYSE",
                Exchange_Timezone="America/New_York", Source="synthetic_fixture",
                Request={"Fixture": "constant_market"}, Retrieved_UTC="2024-01-10T00:00:00Z",
                Imported_UTC="2024-01-10T00:00:00Z", Price_Basis="as_traded",
                Availability_Assumptions=["synthetic, no empirical claims"],
                Redistribution="Synthetic fixture, generated within this project.")


def quote_fixture():
    return pd.DataFrame([dict(Observation_ID="quote_1", Rate_Date="2023-12-29",
                              Available_At=pd.Timestamp("2024-01-01T00:00:00Z"), Quoted_Rate=5.,
                              Rate_Units="percent", Day_Count="ACT/360", Compounding="simple",
                              Source_ID="synthetic")])


def yahoo_fixture():
    index = pd.date_range("2024-01-02", "2024-01-05", tz="America/New_York")
    closes = np.array([50., 49.5, 49.5, 49.5])
    return pd.DataFrame({"Open": closes, "Close": closes, "High": closes+.2, "Low": closes-.2,
                         "Volume": 200., "Adj Close": 49.5, "Dividends": [0., .5, 0., 0.],
                         "Stock Splits": [0., 0., 2., 0.]}, index=index)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[2] / "tmp" / "data_tests"
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bars, self.schedule = market_fixture()

    def save(self, **changes):
        arguments = dict(bars=self.bars, schedule=self.schedule, metadata=metadata_fixture(),
                         cash_policy=CashPolicy(mode="constant_zero"),
                         raw_files={"fixture.csv": b"original fixture bytes\n"}, root=self.root)
        arguments.update(changes)
        return write_snapshot(**arguments)

    def test_roundtrip_is_offline_and_content_addressed(self):
        dataset = self.save()
        with patch("socket.socket", side_effect=AssertionError("network forbidden")):
            loaded = load_dataset(dataset.dataset_id, root=self.root)
        self.assertEqual(len(loaded.bars), len(self.bars))
        self.assertEqual(loaded.bars.Dataset_ID.unique().tolist(), [dataset.dataset_id])
        self.assertEqual(self.save().dataset_id, dataset.dataset_id)
        pd.testing.assert_frame_equal(dataset.bars, loaded.bars)

    def test_new_inputs_preserve_previous_snapshot(self):
        original = self.save()
        changed = self.bars.copy()
        changed.loc[0, "Volume"] += 1
        second = self.save(bars=changed)
        self.assertNotEqual(original.dataset_id, second.dataset_id)
        self.assertEqual(load_dataset(original.dataset_id, root=self.root).bars.Volume.iloc[0], 1000)

    def test_changed_and_missing_files_are_rejected(self):
        dataset = self.save()
        path = self.root / dataset.dataset_id / "bars.csv"
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(DataValidationError, "Checksum"):
            load_dataset(dataset.dataset_id, root=self.root)
        path.write_bytes(original)
        (self.root / dataset.dataset_id / "raw" / "fixture.csv").unlink()
        with self.assertRaises(DataValidationError):
            load_dataset(dataset.dataset_id, root=self.root)

    def test_manifest_changes_and_path_traversal_are_rejected(self):
        dataset = self.save()
        path = self.root / dataset.dataset_id / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["Metadata"]["Source"] = "different_source"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(DataValidationError, "content address"):
            load_dataset(dataset.dataset_id, root=self.root)
        with self.assertRaises(DataValidationError):
            load_dataset("../other", root=self.root)
        with self.assertRaises(DataValidationError):
            self.save(raw_files={"../escape.csv": b"x"})

    def test_invalid_ohlcv_and_duplicate_sessions_fail(self):
        for column, value in [("High", 90), ("Low", 110), ("Close", 0),
                              ("Volume", -1), ("Open", np.nan), ("Volume", np.inf)]:
            with self.subTest(column=column, value=value):
                bars = self.bars.copy()
                bars.loc[0, column] = value
                with self.assertRaises(DataValidationError):
                    self.save(bars=bars)
        duplicate = pd.concat([self.bars.iloc[[0]], self.bars], ignore_index=True)
        with self.assertRaises(DataValidationError):
            self.save(bars=duplicate)

    def test_missing_session_reject_or_report_policy(self):
        partial = self.bars.drop(index=2).reset_index(drop=True)
        with self.assertRaisesRegex(DataValidationError, "missing_sessions"):
            self.save(bars=partial)
        dataset = self.save(bars=partial, validation_policy=ValidationPolicy(missing_sessions="report"))
        self.assertEqual(dataset.quality_report["Missing_Sessions"], 1)
        self.assertIn(self.bars.Session.iloc[2], dataset.quality_report["Warnings"][0]["Dates"])

    def test_incomplete_daily_bar_availability_fails(self):
        bars = self.bars.copy()
        bars.loc[0, "Available_At"] = bars.Open_Time.iloc[0]
        with self.assertRaisesRegex(DataValidationError, "before its close"):
            self.save(bars=bars)

    def test_instrument_metadata_mismatch_is_rejected(self):
        metadata = metadata_fixture()
        metadata["Instrument_ID"] = "different_instrument"
        with self.assertRaisesRegex(DataValidationError, "Instrument metadata"):
            self.save(metadata=metadata)

    def test_missing_required_fields_are_rejected(self):
        with self.assertRaisesRegex(DataValidationError, "missing columns"):
            self.save(bars=self.bars.drop(columns="Volume"))

    def test_provider_missing_requested_final_session_is_detected(self):
        history = yahoo_fixture().iloc[:-1]
        bars, actions, schedule = yahoo_history_tables(history, retrieved_utc="2024-01-06T00:00:00Z",
                                                       end="2024-01-06")
        with self.assertRaisesRegex(DataValidationError, "missing_sessions"):
            self.save(bars=bars, actions=actions, schedule=schedule,
                      validation_policy=ValidationPolicy(dividend_payment="ex_date_credit_approximation"))

    def test_flat_and_zero_volume_history_is_reported(self):
        bars = self.bars.copy()
        bars[["Open", "High", "Low", "Close"]] = 100.
        bars["Volume"] = 0.
        dataset = self.save(bars=bars)
        codes = {row["Code"] for row in dataset.quality_report["Warnings"]}
        self.assertTrue({"zero_volume", "flat_ohlc"}.issubset(codes))

    def test_split_dividend_reconciliation_and_missing_pay_date_policy(self):
        bars, actions, schedule = yahoo_history_tables(yahoo_fixture(), retrieved_utc="2024-01-06T00:00:00Z")
        self.assertEqual(bars.Close.tolist(), [100, 99, 49.5, 49.5])
        self.assertEqual(bars.Volume.tolist(), [100, 100, 200, 200])
        with self.assertRaisesRegex(DataValidationError, "payment date"):
            self.save(bars=bars, actions=actions, schedule=schedule)
        dataset = self.save(bars=bars, actions=actions, schedule=schedule,
                            validation_policy=ValidationPolicy(dividend_payment="ex_date_credit_approximation"))
        np.testing.assert_allclose(dataset.reconciliation.Cash_Total_Return, 0, atol=1e-12)
        np.testing.assert_allclose(dataset.reconciliation.Adjustment_Residual, 0, atol=1e-12)
        self.assertEqual(dataset.quality_report["Return_Reconciliation"]["Intervals"], 3)

    def test_adjusted_prices_plus_duplicate_dividends_fail_reconciliation(self):
        bars, actions, schedule = yahoo_history_tables(yahoo_fixture(), retrieved_utc="2024-01-06T00:00:00Z")
        actions.loc[actions.Action_Type.eq("distribution"), "Amount_Per_Share"] = 5.
        with self.assertRaisesRegex(DataValidationError, "reconciliation failed"):
            self.save(bars=bars, actions=actions, schedule=schedule,
                      validation_policy=ValidationPolicy(dividend_payment="ex_date_credit_approximation"))

    def test_cash_model_is_saved_and_recomputed_offline(self):
        dataset = self.save(cash_policy=CashPolicy(credited_spread_bps=100), cash_observations=quote_fixture())
        self.assertTrue(dataset.cash_intervals.Reference_Return.gt(dataset.cash_intervals.Credited_Return).all())
        loaded = load_dataset(dataset.dataset_id, root=self.root)
        pd.testing.assert_frame_equal(dataset.cash_intervals, loaded.cash_intervals)


class CashTests(unittest.TestCase):
    def test_percentage_units_weekend_and_holiday_accrual(self):
        result = accrue_quoted_rate(5, "2024-05-24T13:30:00Z", "2024-05-28T13:30:00Z")
        self.assertAlmostEqual(result, .05*4/360)
        decimal = accrue_quoted_rate(.05, "2024-05-24T13:30:00Z", "2024-05-28T13:30:00Z", units="decimal")
        self.assertEqual(result, decimal)

    def test_dst_weekend_uses_calendar_days(self):
        result = accrue_quoted_rate(5, "2024-03-08T14:30:00Z", "2024-03-11T13:30:00Z")
        self.assertAlmostEqual(result, .05*3/360)

    def test_compounding_and_credit_spread(self):
        arguments = (5, "2024-01-02T14:30:00Z", "2025-01-02T14:30:00Z")
        self.assertAlmostEqual(accrue_quoted_rate(*arguments, day_count="ACT/365", compounding="annual_effective"),
                               1.05**(366/365)-1)
        self.assertAlmostEqual(accrue_quoted_rate(*arguments, day_count="ACT/365", compounding="continuous"),
                               np.expm1(.05*366/365))
        self.assertAlmostEqual(accrue_quoted_rate(*arguments, spread_bps=100), .04*366/360)

    def test_future_and_stale_quotes_fail(self):
        bars, _ = market_fixture()
        quote = quote_fixture()
        quote["Available_At"] = pd.Timestamp("2024-02-01T00:00:00Z")
        with self.assertRaisesRegex(DataValidationError, "No cash quote"):
            build_cash_intervals(bars, quote, CashPolicy())
        with self.assertRaisesRegex(DataValidationError, "stale"):
            build_cash_intervals(bars, quote_fixture(), CashPolicy(maximum_quote_age_days=1))

    def test_rate_publication_within_interval_waits_until_next_open(self):
        bars, _ = market_fixture()
        quote = quote_fixture()
        newer = quote.copy()
        newer["Observation_ID"] = "quote_2"
        newer["Rate_Date"] = "2024-01-02"
        newer["Available_At"] = bars.Open_Time.iloc[0] + pd.Timedelta(hours=1)
        newer["Quoted_Rate"] = 20.
        intervals = build_cash_intervals(bars, pd.concat([quote, newer]), CashPolicy())
        self.assertEqual(intervals.Observation_IDs.iloc[0], "quote_1")
        self.assertEqual(intervals.Observation_IDs.iloc[1], "quote_2")

    def test_late_older_quote_does_not_replace_newer_reference_date(self):
        bars, _ = market_fixture()
        quote = quote_fixture()
        older = quote.copy()
        older["Observation_ID"] = "late_older_quote"
        older["Rate_Date"] = "2023-12-28"
        older["Available_At"] = bars.Open_Time.iloc[0] - pd.Timedelta(minutes=1)
        older["Quoted_Rate"] = 100.
        intervals = build_cash_intervals(bars, pd.concat([quote, older]), CashPolicy())
        self.assertEqual(set(intervals.Observation_IDs), {"quote_1"})

    def test_fred_missing_values_and_conservative_publication_lag(self):
        observations, excluded = parse_fred_dff(b"observation_date,DFF\n2024-06-18,5.33\n2024-06-19,.\n")
        self.assertEqual(excluded, ["2024-06-19"])
        # Juneteenth is a federal holiday, then Thursday/Friday/Monday are three business days.
        self.assertEqual(observations.Available_At.iloc[0], pd.Timestamp("2024-06-25T03:59:00Z"))
        self.assertEqual(observations.Quoted_Rate.iloc[0], 5.33)

    def test_invalid_and_naive_inputs_fail(self):
        for changes in [{"units": "basis_points"}, {"day_count": "30/360"}, {"compounding": "unknown"}]:
            with self.assertRaises(DataValidationError):
                accrue_quoted_rate(5, "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z", **changes)
        with self.assertRaises(DataValidationError):
            accrue_quoted_rate(5, "2024-01-02", "2024-01-03")

    def test_nonfinite_age_cannot_disable_stale_quote_protection(self):
        for age in [np.nan, np.inf, -1, 1.5, True]:
            with self.subTest(age=age), self.assertRaises(ValueError):
                CashPolicy(maximum_quote_age_days=age)


class CalendarAndEligibilityTests(unittest.TestCase):
    def test_holiday_early_close_and_dst(self):
        schedule = exchange_schedule("2024-07-03", "2024-07-05")
        self.assertNotIn("2024-07-04", schedule.Session.tolist())
        self.assertEqual(schedule.Close_Time.iloc[0], pd.Timestamp("2024-07-03T17:00:00Z"))
        schedule = exchange_schedule("2024-03-08", "2024-03-11")
        self.assertEqual(schedule.Open_Time.iloc[0].hour, 14)
        self.assertEqual(schedule.Open_Time.iloc[1].hour, 13)

    def test_future_split_restoration_preserves_prior_executable_prices(self):
        after = restore_as_traded_history(yahoo_fixture())
        before = yahoo_fixture().iloc[:2].copy()
        before[["Open", "High", "Low", "Close", "Dividends"]] *= 2
        before["Volume"] /= 2
        before = restore_as_traded_history(before)
        pd.testing.assert_frame_equal(before[["Open", "High", "Low", "Close", "Volume", "Dividends"]],
                                      after.iloc[:2][["Open", "High", "Low", "Close", "Volume", "Dividends"]])

    def test_purged_history_eligibility_has_no_invented_preinception_rows(self):
        bars, schedule = market_fixture("2022-01-03", "2023-12-29")
        from regime_research.data.contracts import Dataset
        dataset = Dataset(bars, empty_table("actions"), empty_table("cash_observations"),
                          empty_table("cash_intervals"), schedule, {"Dataset_ID": "test"}, {})
        requirements = HistoryRequirements(training_years=1, feature_warmup_sessions=5,
                                            calibration_sessions=10, minimum_training_rows=20)
        sample = build_eligible_sample(dataset, requirements)
        self.assertGreater(len(sample.eligible), 100)
        first_usable = pd.Timestamp(bars.Session.iloc[4])
        self.assertGreaterEqual(pd.Timestamp(sample.eligible.Session.iloc[0]), first_usable + pd.DateOffset(years=1))
        self.assertTrue(sample.eligible.Training_Label_End.lt(sample.eligible.Calibration_Start).all())
        self.assertTrue(sample.eligible.Calibration_Label_End.lt(sample.eligible.Decision_Time).all())
        self.assertTrue(sample.eligible.Execution_Time.lt(sample.eligible.Return_End_Time).all())
        self.assertEqual(len(sample.eligible) + len(sample.exclusions), len(bars))
        self.assertIn("unrealized_forward_return", sample.exclusions.Reasons.iloc[-1])
        impossible = build_eligible_sample(dataset, HistoryRequirements())
        self.assertEqual(len(impossible.eligible), 0)

    def test_history_gap_disqualifies_window(self):
        bars, schedule = market_fixture("2022-01-03", "2023-12-29")
        from regime_research.data.contracts import Dataset
        bars = bars.drop(index=300).reset_index(drop=True)
        dataset = Dataset(bars, empty_table("actions"), empty_table("cash_observations"),
                          empty_table("cash_intervals"), schedule, {"Dataset_ID": "test"}, {})
        sample = build_eligible_sample(dataset, HistoryRequirements(training_years=1,
                                        feature_warmup_sessions=5, calibration_sessions=10, minimum_training_rows=20))
        later = sample.exclusions.loc[sample.exclusions.Session.gt("2023-05-01")]
        self.assertTrue(later.Reasons.str.contains("missing_exchange_session").all())

    def test_future_price_perturbations_preserve_prior_normalized_bars(self):
        history = yahoo_fixture()
        original, _, _ = yahoo_history_tables(history, retrieved_utc="2024-01-06T00:00:00Z")
        history.loc[history.index[-1], ["Open", "High", "Low", "Close", "Adj Close"]] *= 2
        changed, _, _ = yahoo_history_tables(history, retrieved_utc="2024-01-06T00:00:00Z")
        pd.testing.assert_frame_equal(original.iloc[:-1], changed.iloc[:-1])


if __name__ == "__main__":
    unittest.main()

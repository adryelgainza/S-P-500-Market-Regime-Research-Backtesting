"""Specific regressions for financing, entitlement timing, and return boundaries."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from regime_research.data import DataValidationError, empty_table, exchange_schedule, normalize_bars
from regime_research.execution import (ExecutionConfig, portfolio_metrics, save_portfolio_result,
                                       simulate_portfolio, solve_target_trade, verify_portfolio_result)


def inputs(prices, weights, *, policy="daily_target", cash_rate=0.0):
    schedule = exchange_schedule("2024-01-02", "2024-04-30").iloc[:len(prices)].reset_index(drop=True)
    raw = pd.DataFrame(dict(Session=schedule.Session, Open=prices, High=np.asarray(prices)*1.01,
                            Low=np.asarray(prices)*.99, Close=prices, Volume=1000.))
    bars = normalize_bars(raw, schedule, instrument="TEST", source="synthetic")
    intervals = pd.DataFrame(dict(Start_Time=bars.Open_Time.iloc[:-1].to_list(),
                                  End_Time=bars.Open_Time.iloc[1:].to_list(),
                                  Credited_Return=cash_rate, Reference_Return=cash_rate))
    targets = pd.DataFrame(dict(Strategy_ID="test", Rule_Version="v1", Target_Weight=weights,
                                Decision_Time=bars.Available_At.iloc[:len(weights)].to_list(),
                                Execution_Time=bars.Open_Time.iloc[1:len(weights)+1].to_list(),
                                Rebalance_Policy=policy))
    return bars, empty_table("actions"), intervals, targets


def action(bars, index, kind, amount, *, payment=None, name="action"):
    return dict(Instrument_ID="TEST", Action_ID=name, Action_Type=kind,
                Effective_Time=bars.Open_Time.iloc[index], Ex_Date=bars.Session.iloc[index],
                Pay_Date=payment if payment is not None else pd.NA,
                Amount_Per_Share=amount if kind == "distribution" else np.nan,
                Split_Ratio=amount if kind == "split" else np.nan,
                Available_At=bars.Open_Time.iloc[index], Source_ID="synthetic")


def reference_wealth(prices, weights, rates, cost, initial=100.0, liquidate=False):
    """Independent no-action ledger using share-space bisection for each fill."""
    cash, shares = initial, 0.0
    before = []
    for index, weight in enumerate(weights):
        price = prices[index]
        before.append(cash + shares*price)
        low, high = 0.0, shares + cash/(price*(1+cost))
        for _ in range(80):
            candidate = (low+high)/2
            candidate_cost = abs(candidate-shares)*price*cost
            candidate_cash = cash - (candidate-shares)*price - candidate_cost
            realized = candidate*price/(candidate*price+candidate_cash)
            if realized < weight:
                low = candidate
            else:
                high = candidate
        next_shares = (low+high)/2
        cash -= (next_shares-shares)*price + abs(next_shares-shares)*price*cost
        shares = next_shares
        cash *= 1+rates[index]
    terminal = cash + shares*prices[-1]
    if liquidate:
        terminal -= shares*prices[-1]*cost
    before.append(terminal)
    return np.asarray(before)


class FinancingTests(unittest.TestCase):
    def test_all_cash_tracks_cash_returns_and_excess_is_zero(self):
        result = simulate_portfolio(*inputs([100]*5, [0]*3, cash_rate=.01), ExecutionConfig(initial_capital=100))
        self.assertAlmostEqual(result.daily.End_Wealth.iloc[-1], 100*1.01**3)
        np.testing.assert_allclose(result.daily.Net_Excess_Return, 0, atol=1e-15)
        self.assertEqual(len(result.trades), 0)

    def test_entry_is_funded_after_cost_and_initial_drawdown_includes_cost(self):
        result = simulate_portfolio(*inputs([100]*4, [1, 1]), ExecutionConfig(initial_capital=100, one_way_cost=.01))
        self.assertAlmostEqual(result.trades.Cost.sum(), 100/101)
        self.assertAlmostEqual(result.daily.End_Wealth.iloc[-1], 100/1.01)
        self.assertAlmostEqual(result.daily.Drawdown.iloc[0], 1/1.01-1)
        self.assertAlmostEqual(result.valuations.Cash.iloc[0], 0)

    def test_constant_half_target_rebalances_drift(self):
        result = simulate_portfolio(*inputs([100, 100, 110, 110], [.5, .5]), ExecutionConfig(initial_capital=100, one_way_cost=0))
        self.assertAlmostEqual(result.valuations.Pre_Trade_Wealth.iloc[1], 105)
        self.assertAlmostEqual(result.trades.Signed_Notional.iloc[1], -2.5)
        self.assertAlmostEqual(result.valuations.Realized_Weight.iloc[1], .5)

    def test_signal_change_policy_preserves_drift_and_allows_sparse_signals(self):
        args = list(inputs([100, 100, 110, 110], [.5, .5], policy="signal_change_only"))
        args[3] = args[3].iloc[:1]
        result = simulate_portfolio(*args, ExecutionConfig(initial_capital=100, one_way_cost=0))
        self.assertEqual(len(result.trades), 1)
        self.assertAlmostEqual(result.valuations.Realized_Weight.iloc[1], 55/105)

    def test_strategy_switch_uses_continuing_wealth_and_costs(self):
        args = list(inputs([100, 100, 110, 110], [1, 0]))
        args[3].loc[1, "Rule_Version"] = "new_recipe"
        result = simulate_portfolio(*args, ExecutionConfig(initial_capital=100, one_way_cost=.01))
        self.assertEqual(result.trades.Trade_Type.tolist(), ["entry", "strategy_switch"])
        self.assertAlmostEqual(result.daily.End_Wealth.iloc[-1], 100/1.01*1.1*.99)
        self.assertAlmostEqual(result.daily.Net_Return.iloc[0], 1.1/1.01-1)
        self.assertAlmostEqual(result.daily.Net_Return.iloc[1], -.01)

    def test_known_terminal_liquidation_cost_is_in_final_period_once(self):
        args = inputs([100]*4, [1, 1])
        result = simulate_portfolio(*args, ExecutionConfig(initial_capital=100, one_way_cost=.01, terminal_liquidation=True))
        self.assertAlmostEqual(result.daily.End_Wealth.iloc[-1], 100/1.01*.99)
        self.assertAlmostEqual(result.daily.Terminal_Cost.sum(), (100/1.01)*.01)
        self.assertAlmostEqual(result.daily.Transaction_Cost.sum(), result.trades.Cost.sum())
        self.assertEqual(result.valuations.Shares.iloc[-1], 0)

    def test_zero_cost_liquidation_still_records_turnover(self):
        result = simulate_portfolio(*inputs([100]*4, [1, 1]), ExecutionConfig(one_way_cost=0, terminal_liquidation=True))
        self.assertAlmostEqual(result.daily.Turnover.sum(), 2)

    def test_terminal_cost_fractions_use_wealth_at_the_terminal_trade(self):
        result = simulate_portfolio(*inputs([100, 100, 100, 200], [1, 1]),
                                    ExecutionConfig(one_way_cost=.01, terminal_liquidation=True))
        self.assertAlmostEqual(portfolio_metrics(result)["Cost_Equity_Fractions"], .01/1.01+.01)

    def test_solver_respects_unsettled_receivables(self):
        notional, cost, weight, shortfall = solve_target_trade(50, 0, 50, 1, .01)
        self.assertEqual(notional, 0)
        self.assertEqual(cost, 0)
        self.assertEqual(weight, .5)
        self.assertEqual(shortfall, .5)

    def test_random_paths_agree_with_independent_share_bisection_ledger(self):
        rng = np.random.default_rng(2841)
        for _ in range(60):
            prices = 100*np.exp(np.cumsum(rng.normal(0, .06, 9)))
            weights = rng.choice([0, .2, .5, .8, 1.], 7)
            rate = float(rng.choice([0, .0002, -.0001]))
            cost = float(rng.choice([0, .0005, .01]))
            liquidate = bool(rng.integers(2))
            result = simulate_portfolio(*inputs(prices, weights, cash_rate=rate),
                                         ExecutionConfig(initial_capital=100, one_way_cost=cost, terminal_liquidation=liquidate))
            reference = reference_wealth(prices[1:], weights, [rate]*7, cost, liquidate=liquidate)
            np.testing.assert_allclose(result.daily.End_Wealth, reference[1:], rtol=1e-8)
            self.assertLess(abs(result.accounting_reconciliation["Pnl_Bridge_Error"]), 1e-10)


class CorporateActionTests(unittest.TestCase):
    def test_ex_date_buyer_gets_no_entitlement(self):
        args = list(inputs([100, 99, 99, 99], [1, 1]))
        args[1] = pd.DataFrame([action(args[0], 1, "distribution", 1, payment=args[0].Session.iloc[2])])
        result = simulate_portfolio(*args, ExecutionConfig(initial_capital=99, one_way_cost=0))
        self.assertEqual(result.daily.Distribution_PnL.sum(), 0)
        self.assertEqual(len(result.cash_events), 0)

    def test_sale_at_ex_open_keeps_entitlement_and_later_payment(self):
        args = list(inputs([100, 100, 99, 99, 99], [1, 0, 0]))
        args[1] = pd.DataFrame([action(args[0], 2, "distribution", 1, payment=args[0].Session.iloc[3])])
        result = simulate_portfolio(*args, ExecutionConfig(initial_capital=100, one_way_cost=0))
        self.assertEqual(result.valuations.Receivables.iloc[1], 1)
        self.assertEqual(result.valuations.Cash.iloc[1], 99)
        self.assertEqual(result.valuations.Cash.iloc[2], 100)
        self.assertEqual(result.daily.End_Wealth.iloc[-1], 100)
        self.assertEqual(result.daily.Distribution_PnL.sum(), 1)

    def test_unpaid_receivable_earns_no_interest(self):
        args = list(inputs([100, 100, 99, 99], [1, 1], cash_rate=.1))
        args[1] = pd.DataFrame([action(args[0], 2, "distribution", 1, payment="2024-02-01")])
        result = simulate_portfolio(*args, ExecutionConfig(initial_capital=100, one_way_cost=0))
        self.assertEqual(result.daily.Cash_Interest.sum(), 0)
        self.assertEqual(result.accounting_reconciliation["Outstanding_Receivables"], 1)

    def test_cash_payment_inside_interval_receives_only_remaining_interest(self):
        args = list(inputs([100, 100, 99, 99, 99], [1, 0, 0], cash_rate=.1))
        # Entitlement at Thursday's open, payment Friday midnight before its open.
        args[1] = pd.DataFrame([action(args[0], 2, "distribution", 1, payment=args[0].Session.iloc[3])])
        result = simulate_portfolio(*args, ExecutionConfig(initial_capital=100, one_way_cost=0))
        self.assertAlmostEqual(result.daily.Cash_Interest.iloc[1], 99*.1 + .1*(9.5/24))

    def test_split_and_same_day_dividend_use_post_split_shares(self):
        args = list(inputs([100, 100, 49.5, 49.5], [1, 1]))
        args[1] = pd.DataFrame([action(args[0], 2, "split", 2, name="split"),
                                action(args[0], 2, "distribution", .5, payment="2024-02-01", name="dividend")])
        result = simulate_portfolio(*args, ExecutionConfig(initial_capital=100, one_way_cost=0))
        self.assertAlmostEqual(result.valuations.Shares.iloc[1], 2)
        self.assertAlmostEqual(result.valuations.Receivables.iloc[1], 1)
        self.assertAlmostEqual(result.daily.End_Wealth.iloc[-1], 100)

    def test_split_does_not_change_an_existing_receivable(self):
        args = list(inputs([100, 100, 99, 49.5, 49.5], [1, 1, 1]))
        args[1] = pd.DataFrame([action(args[0], 2, "distribution", 1, payment="2024-02-01", name="dividend"),
                                action(args[0], 3, "split", 2, name="split")])
        result = simulate_portfolio(*args, ExecutionConfig(initial_capital=100, one_way_cost=0))
        self.assertEqual(result.valuations.Receivables.iloc[-1], 1)
        self.assertEqual(result.daily.End_Wealth.iloc[-1], 100)

    def test_payment_on_closed_session_is_processed_before_next_open(self):
        args = list(inputs([100]*7, [1, 1, 0, 0, 0]))
        args[1] = pd.DataFrame([action(args[0], 3, "distribution", 1, payment="2024-01-06")])
        result = simulate_portfolio(*args, ExecutionConfig(initial_capital=100, one_way_cost=0))
        payment = result.cash_events.loc[result.cash_events.Event_Type.eq("distribution_payment")].iloc[0]
        self.assertEqual(payment.Event_Time, pd.Timestamp("2024-01-06T05:00:00Z"))
        self.assertEqual(result.accounting_reconciliation["Outstanding_Receivables"], 0)

    def test_missing_payment_date_requires_and_records_approximation(self):
        args = list(inputs([100, 100, 99, 99], [1, 1]))
        args[1] = pd.DataFrame([action(args[0], 2, "distribution", 1)])
        with self.assertRaises(DataValidationError):
            simulate_portfolio(*args)
        result = simulate_portfolio(*args, ExecutionConfig(initial_capital=100, one_way_cost=0,
                                    dividend_payment_policy="ex_date_credit_approximation"))
        self.assertEqual(result.accounting_reconciliation["Approximated_Distribution_Events"], 1)
        self.assertEqual(result.valuations.Receivables.sum(), 0)


class ExecutionContractTests(unittest.TestCase):
    def test_timestamp_configuration_is_serializable_and_currency_is_checked(self):
        args = inputs([100]*4, [1, 1])
        config = ExecutionConfig(end_time=args[0].Open_Time.iloc[-1])
        result = simulate_portfolio(*args, config)
        json.dumps(result.accounting_reconciliation)
        args[0]["Currency"] = "EUR"
        with self.assertRaises(DataValidationError):
            simulate_portfolio(*args)

    def test_invalid_targets_and_late_decisions_fail(self):
        for weight in [-.1, 1.1, np.nan, np.inf]:
            with self.subTest(weight=weight), self.assertRaises(DataValidationError):
                simulate_portfolio(*inputs([100]*4, [weight, 0]))
        args = list(inputs([100]*4, [1, 1]))
        args[3].loc[0, "Decision_Time"] = args[3].Execution_Time.iloc[0]
        with self.assertRaises(DataValidationError):
            simulate_portfolio(*args)

    def test_missing_daily_target_and_cash_interval_fail(self):
        args = list(inputs([100]*5, [.5]*3))
        args[3] = args[3].drop(index=1)
        with self.assertRaisesRegex(DataValidationError, "missing an execution"):
            simulate_portfolio(*args)
        args = list(inputs([100]*4, [1, 1]))
        args[2] = args[2].iloc[:1]
        with self.assertRaisesRegex(DataValidationError, "Cash intervals"):
            simulate_portfolio(*args)

    def test_parquet_outputs_are_published_once_with_checksums(self):
        result = simulate_portfolio(*inputs([100]*4, [.5, .5]))
        base = Path(__file__).resolve().parents[2] / "tmp" / "execution_tests"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            output = save_portfolio_result(result, Path(directory)/"result")
            pd.testing.assert_frame_equal(result.daily, pd.read_parquet(output/"portfolio_daily.parquet"))
            manifest = json.loads((output/"completion.json").read_text())
            self.assertIn("accounting_reconciliation.json", manifest["Files"])
            self.assertEqual(verify_portfolio_result(output)["Periods"], 2)
            with self.assertRaises(FileExistsError):
                save_portfolio_result(result, output)
            trade_path = output/"trades.parquet"
            trade_path.write_bytes(trade_path.read_bytes()+b"damage")
            with self.assertRaisesRegex(DataValidationError, "checksum mismatch"):
                verify_portfolio_result(output)
        self.assertEqual(portfolio_metrics(result)["Periods"], 2)


if __name__ == "__main__":
    unittest.main()

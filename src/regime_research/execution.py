"""Self-financing, next-open execution for fractional-share long/cash portfolios."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .data.contracts import ACTION_COLUMNS, DataValidationError, utc_timestamp
from .experiments import publish_directory


TARGET_COLUMNS = ["Strategy_ID", "Decision_Time", "Execution_Time", "Target_Weight",
                  "Rebalance_Policy", "Rule_Version"]
TRADE_COLUMNS = ["Execution_Time", "Decision_Time", "Strategy_ID", "Rule_Version", "Trade_Type",
                 "Price", "Shares_Traded", "Signed_Notional", "Cost", "Turnover",
                 "Target_Weight", "Realized_Weight", "Target_Weight_Shortfall"]
EVENT_COLUMNS = ["Event_Time", "Event_Type", "Action_ID", "Amount", "Cash_Delta",
                 "Receivable_Delta", "Shares_Delta"]


@dataclass(frozen=True)
class ExecutionConfig:
    initial_capital: float = 1.0
    one_way_cost: float = 0.0005
    dividend_payment_policy: str = "reject"
    exchange_timezone: str = "America/New_York"
    terminal_liquidation: bool = False
    end_time: str | None = None
    relative_tolerance: float = 1e-12

    def __post_init__(self):
        if not np.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("Initial capital must be finite and positive")
        if not np.isfinite(self.one_way_cost) or not 0 <= self.one_way_cost < 1:
            raise ValueError("One-way cost must be between zero and one")
        if self.dividend_payment_policy not in {"reject", "ex_date_credit_approximation"}:
            raise ValueError("Choose an explicit dividend payment policy")
        if not 0 < self.relative_tolerance <= 1e-8:
            raise ValueError("Invalid accounting tolerance")
        if not isinstance(self.terminal_liquidation, bool):
            raise ValueError("terminal_liquidation must be boolean")
        if self.end_time is not None:
            object.__setattr__(self, "end_time", utc_timestamp(self.end_time).isoformat())


@dataclass(frozen=True)
class PortfolioResult:
    daily: pd.DataFrame
    valuations: pd.DataFrame
    trades: pd.DataFrame
    cash_events: pd.DataFrame
    targets: pd.DataFrame
    accounting_reconciliation: dict


def solve_target_trade(equity_value, cash, receivables, target_weight, one_way_cost):
    """Solve the post-cost target, then apply settled-cash funding constraints."""
    values = np.asarray([equity_value, cash, receivables, target_weight, one_way_cost], dtype=float)
    if (not np.isfinite(values).all() or min(equity_value, cash, receivables) < 0
            or not 0 <= target_weight <= 1 or not 0 <= one_way_cost < 1):
        raise DataValidationError("Invalid long/cash trade inputs")
    wealth = equity_value + cash + receivables
    if wealth <= 0:
        raise DataValidationError("Portfolio wealth must be positive")
    desired = target_weight * wealth - equity_value
    denominator = 1 + target_weight * one_way_cost if desired >= 0 else 1 - target_weight * one_way_cost
    unconstrained_notional = desired / denominator
    notional = float(np.clip(unconstrained_notional, -equity_value, cash / (1 + one_way_cost)))
    cost = one_way_cost * abs(notional)
    realized_weight = (equity_value + notional) / (wealth - cost)
    return notional, cost, realized_weight, max(0.0, target_weight - realized_weight)


def _timestamp_columns(frame, columns):
    frame = frame.copy().reset_index(drop=True)
    for column in columns:
        if column not in frame:
            raise DataValidationError(f"Missing timestamp column {column}")
        frame[column] = pd.DatetimeIndex([utc_timestamp(value) for value in frame[column]], tz="UTC")
    return frame


def _prepare_inputs(bars, actions, cash_intervals, targets, config):
    required_bars = {"Session", "Instrument_ID", "Currency", "Open", "Open_Time", "Close_Time", "Available_At"}
    if not required_bars.issubset(bars) or len(bars) < 3:
        raise DataValidationError("Execution needs market bars and a preceding decision-session context")
    bars = _timestamp_columns(bars, ["Open_Time", "Close_Time", "Available_At"])
    if (bars.Session.isna().any() or bars.Session.duplicated().any() or not bars.Open_Time.is_monotonic_increasing
            or bars.Open_Time.duplicated().any() or bars.Instrument_ID.isna().any() or bars.Instrument_ID.nunique() != 1):
        raise DataValidationError("Execution bars must be ordered, unique, and single-instrument")
    if not bars.Currency.eq("USD").all():
        raise DataValidationError("This execution engine requires USD-denominated bars")
    if not bars.Open_Time.dt.tz_convert(config.exchange_timezone).dt.strftime("%Y-%m-%d").eq(bars.Session).all():
        raise DataValidationError("Execution opens disagree with their local session labels")
    if not np.isfinite(bars.Open.to_numpy(float)).all() or bars.Open.le(0).any():
        raise DataValidationError("Execution prices must be finite and positive")
    if not (bars.Open_Time.lt(bars.Close_Time) & bars.Close_Time.le(bars.Available_At)).all():
        raise DataValidationError("Invalid market-bar availability chronology")
    if not set(TARGET_COLUMNS).issubset(targets) or targets.empty:
        raise DataValidationError("Targets require the complete target schema and an initial decision")
    targets = _timestamp_columns(targets, ["Decision_Time", "Execution_Time"])
    if targets.Execution_Time.duplicated().any() or not targets.Execution_Time.is_monotonic_increasing:
        raise DataValidationError("Targets must have unique, ordered execution times")
    weights = targets.Target_Weight.to_numpy(float)
    if not np.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any():
        raise DataValidationError("Target weights must be finite and between zero and one")
    for column in ["Strategy_ID", "Rule_Version", "Rebalance_Policy"]:
        if targets[column].isna().any() or targets[column].astype(str).str.strip().eq("").any():
            raise DataValidationError(f"Targets require {column}")
    if not targets.Rebalance_Policy.isin(["daily_target", "signal_change_only"]).all():
        raise DataValidationError("Unsupported rebalance policy")
    opens = pd.DatetimeIndex(bars.Open_Time)
    indices = opens.get_indexer(targets.Execution_Time)
    if (indices < 1).any():
        raise DataValidationError("Each target needs an execution open and its preceding session in the bars")
    prior_availability = pd.DatetimeIndex(bars.Available_At.iloc[indices - 1])
    if not (targets.Decision_Time.ge(prior_availability)
            & targets.Decision_Time.lt(targets.Execution_Time)).all():
        raise DataValidationError("A target must use a completed preceding bar before the next open")
    start = int(indices[0])
    end_time = utc_timestamp(config.end_time) if config.end_time is not None else opens[-1]
    end = opens.get_indexer([end_time])[0]
    if end <= start or (indices >= end).any():
        raise DataValidationError("The terminal valuation must follow every target execution")
    intervals = _timestamp_columns(cash_intervals, ["Start_Time", "End_Time"])
    if not {"Reference_Return", "Credited_Return"}.issubset(intervals):
        raise DataValidationError("Cash intervals require reference and credited returns")
    if intervals.Start_Time.duplicated().any():
        raise DataValidationError("Duplicate cash intervals")
    interval_map = {row.Start_Time: row for row in intervals.itertuples(index=False)}
    for index in range(start, end):
        row = interval_map.get(opens[index])
        if row is None or row.End_Time != opens[index + 1]:
            raise DataValidationError("Cash intervals must match each execution interval exactly")
        if not np.isfinite([row.Reference_Return, row.Credited_Return]).all() or min(row.Reference_Return, row.Credited_Return) <= -1:
            raise DataValidationError("Invalid cash interval return")
    if not set(ACTION_COLUMNS).issubset(actions):
        raise DataValidationError("Corporate actions require the Layer 1 action schema")
    actions = _timestamp_columns(actions, ["Effective_Time", "Available_At"])
    if actions.Action_ID.isna().any() or actions.Action_ID.duplicated().any():
        raise DataValidationError("Action IDs must be unique and present")
    open_by_session = dict(zip(bars.Session, bars.Open_Time))
    relevant_actions = actions.loc[actions.Effective_Time.between(opens[start], opens[end])].copy()
    for action in relevant_actions.itertuples(index=False):
        if (action.Instrument_ID != bars.Instrument_ID.iloc[0]
                or open_by_session.get(action.Ex_Date) != action.Effective_Time):
            raise DataValidationError("Corporate actions must match the instrument and ex-date open")
        if action.Action_Type == "split":
            if not np.isfinite(action.Split_Ratio) or action.Split_Ratio <= 0:
                raise DataValidationError("Invalid split ratio")
        elif action.Action_Type == "distribution":
            if not np.isfinite(action.Amount_Per_Share) or action.Amount_Per_Share < 0:
                raise DataValidationError("Invalid distribution amount")
            if pd.isna(action.Pay_Date) or action.Pay_Date == "":
                if config.dividend_payment_policy == "reject":
                    raise DataValidationError("Missing dividend payment date requires an explicit approximation")
            elif str(action.Pay_Date) != pd.Timestamp(action.Pay_Date).strftime("%Y-%m-%d") or action.Pay_Date < action.Ex_Date:
                raise DataValidationError("Invalid distribution payment date")
        else:
            raise DataValidationError("Unsupported corporate action")
    return bars.iloc[start:end+1].reset_index(drop=True), relevant_actions, interval_map, targets


def _settle_interval(cash, pending, start, end, credited_return, timezone, events):
    """Accrue simple interval interest on settled principal between payments."""
    local_start = start.tz_convert(timezone).tz_localize(None)
    duration = (end.tz_convert(timezone).tz_localize(None) - local_start).total_seconds()
    interest = cash * credited_return
    remaining = []
    for payment in sorted(pending, key=lambda item: (item["Time"], item["Action_ID"])):
        if payment["Time"] > end:
            remaining.append(payment)
            continue
        if payment["Time"] < start:
            raise DataValidationError("Overdue receivable encountered during accrual")
        elapsed = (payment["Time"].tz_convert(timezone).tz_localize(None) - local_start).total_seconds()
        fraction_remaining = float(np.clip(1 - elapsed / duration, 0, 1))
        amount = payment["Amount"]
        interest += amount * credited_return * fraction_remaining
        cash += amount
        events.append(dict(Event_Time=payment["Time"], Event_Type="distribution_payment",
                           Action_ID=payment["Action_ID"], Amount=amount, Cash_Delta=amount,
                           Receivable_Delta=-amount, Shares_Delta=0.0))
    cash += interest
    if interest != 0:
        events.append(dict(Event_Time=end, Event_Type="cash_interest", Action_ID="",
                           Amount=interest, Cash_Delta=interest, Receivable_Delta=0.0, Shares_Delta=0.0))
    return cash, remaining, interest


def simulate_portfolio(bars, actions, cash_intervals, target_weights,
                       execution_config: ExecutionConfig | None = None) -> PortfolioResult:
    """Start at the first target's open and mark through the requested final open.

    Include the preceding decision session in bars. Daily targets must cover
    every tradable interval. Sparse signal-change targets carry their holdings
    until the next supplied signal. Inputs use explicit event-time conventions.
    """
    config = execution_config or ExecutionConfig()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    bars, actions, intervals, targets = _prepare_inputs(bars, actions, cash_intervals, target_weights, config)
    target_map = {row.Execution_Time: row for row in targets.itertuples(index=False)}
    action_groups = {time: frame for time, frame in actions.groupby("Effective_Time")}
    cash, shares = config.initial_capital, 0.0
    pending, states, trades, events = [], [], [], []
    active = None
    maximum_identity_error = 0.0
    approximated_distributions = 0
    for index, bar in enumerate(bars.itertuples(index=False)):
        time, price = bar.Open_Time, float(bar.Open)
        price_pnl = distribution_pnl = interest = 0.0
        if index:
            prior = states[-1]
            interval = intervals[prior["Valuation_Time"]]
            cash, pending, interest = _settle_interval(cash, pending, prior["Valuation_Time"], time,
                                                       interval.Credited_Return, config.exchange_timezone, events)
        actions_now = action_groups.get(time)
        if actions_now is not None:
            for action in actions_now.loc[actions_now.Action_Type.eq("split")].itertuples(index=False):
                shares_delta = shares * (action.Split_Ratio - 1)
                shares *= action.Split_Ratio
                events.append(dict(Event_Time=time, Event_Type="split", Action_ID=action.Action_ID,
                                   Amount=action.Split_Ratio, Cash_Delta=0.0,
                                   Receivable_Delta=0.0, Shares_Delta=shares_delta))
            for action in actions_now.loc[actions_now.Action_Type.eq("distribution")].itertuples(index=False):
                missing_date = pd.isna(action.Pay_Date) or action.Pay_Date == ""
                approximated_distributions += int(missing_date)
                amount = shares * action.Amount_Per_Share
                if amount == 0:
                    continue
                distribution_pnl += amount
                payment_time = time if missing_date else max(time, pd.Timestamp(action.Pay_Date)
                                                             .tz_localize(config.exchange_timezone).tz_convert("UTC"))
                events.append(dict(Event_Time=time, Event_Type="distribution_entitlement", Action_ID=action.Action_ID,
                                   Amount=amount, Cash_Delta=0.0, Receivable_Delta=amount, Shares_Delta=0.0))
                if payment_time == time:
                    cash += amount
                    events.append(dict(Event_Time=time, Event_Type="distribution_payment", Action_ID=action.Action_ID,
                                       Amount=amount, Cash_Delta=amount, Receivable_Delta=-amount, Shares_Delta=0.0))
                else:
                    pending.append(dict(Time=payment_time, Action_ID=action.Action_ID, Amount=amount))
        receivables = sum(payment["Amount"] for payment in pending)
        equity_value = shares * price
        before = equity_value + cash + receivables
        shares_before, cash_before = shares, cash
        tolerance = config.relative_tolerance * max(before, config.initial_capital)
        if index:
            price_pnl = equity_value - prior["Shares"] * prior["Price"]
            expected = prior["Post_Trade_Wealth"] + price_pnl + distribution_pnl + interest
            maximum_identity_error = max(maximum_identity_error, abs(before - expected) / max(before, expected))
            if abs(before - expected) > tolerance:
                raise ArithmeticError("Open-to-open wealth identity failed")
        if before <= 0 or not np.isfinite(before):
            raise ArithmeticError("Portfolio wealth must remain finite and positive")
        target = target_map.get(time)
        changed_signal = target is not None and (active is None or any(
            getattr(target, key) != getattr(active, key)
            for key in ["Strategy_ID", "Rule_Version", "Target_Weight", "Rebalance_Policy"]))
        if target is not None:
            active = target
        terminal = index == len(bars) - 1
        if not terminal and target is None and active.Rebalance_Policy == "daily_target":
            raise DataValidationError("A daily-target strategy is missing an execution-session target")
        should_trade = not terminal and (active.Rebalance_Policy == "daily_target" or changed_signal)
        trade_type = "entry" if not trades else "rebalance"
        if states and changed_signal and (active.Strategy_ID, active.Rule_Version) != (prior["Strategy_ID"], prior["Rule_Version"]):
            trade_type = "strategy_switch"
        notional = cost = shortfall = 0.0
        realized = equity_value / before
        desired_weight = float(active.Target_Weight)
        if terminal and config.terminal_liquidation:
            should_trade, desired_weight, trade_type = True, 0.0, "terminal_liquidation"
        if should_trade:
            notional, cost, realized, shortfall = solve_target_trade(equity_value, cash, receivables,
                                                                   desired_weight, config.one_way_cost)
            if abs(notional) <= tolerance:
                notional = cost = 0.0
                realized = equity_value / before
                shortfall = max(0.0, desired_weight - realized)
            else:
                shares += notional / price
                cash -= notional + cost
                trades.append(dict(Execution_Time=time, Decision_Time=active.Decision_Time,
                                   Strategy_ID=active.Strategy_ID, Rule_Version=active.Rule_Version,
                                   Trade_Type=trade_type, Price=price, Shares_Traded=notional/price,
                                   Signed_Notional=notional, Cost=cost, Turnover=abs(notional)/before,
                                   Target_Weight=desired_weight, Realized_Weight=realized,
                                   Target_Weight_Shortfall=shortfall))
        if cash < -tolerance or shares * price < -tolerance:
            raise ArithmeticError("A trade created negative cash or shares")
        cash, shares = max(0.0, cash), max(0.0, shares)
        after = shares * price + cash + receivables
        if abs(after - (before - cost)) > tolerance:
            raise ArithmeticError("Trade financing identity failed")
        states.append(dict(Valuation_Time=time, Session=bar.Session, Price=price,
                           Pre_Trade_Wealth=before, Post_Trade_Wealth=after, Shares=shares,
                           Pre_Trade_Shares=shares_before, Pre_Trade_Cash=cash_before,
                           Pre_Trade_Weight=equity_value/before,
                           Cash=cash, Receivables=receivables, Target_Weight=desired_weight,
                           Realized_Weight=shares*price/after, Signed_Notional=notional, Cost=cost,
                           Turnover=abs(notional)/before, Target_Weight_Shortfall=shortfall,
                           Strategy_ID=active.Strategy_ID, Rule_Version=active.Rule_Version,
                           Rebalance_Policy=active.Rebalance_Policy, Price_PnL=price_pnl,
                           Distribution_PnL=distribution_pnl, Cash_Interest=interest,
                           Terminal_Cost=cost if terminal else 0.0,
                           Terminal_Liquidation=terminal and config.terminal_liquidation))
    rows = []
    for start, end in zip(states[:-1], states[1:]):
        reference = intervals[start["Valuation_Time"]].Reference_Return
        total_cost = start["Cost"] + end["Terminal_Cost"]
        gross_pnl = end["Price_PnL"] + end["Distribution_PnL"] + end["Cash_Interest"]
        ending_wealth = end["Pre_Trade_Wealth"] - end["Terminal_Cost"]
        net_return = ending_wealth / start["Pre_Trade_Wealth"] - 1
        rows.append(dict(Execution_Time=start["Valuation_Time"], Return_End_Time=end["Valuation_Time"],
                         Strategy_ID=start["Strategy_ID"], Rule_Version=start["Rule_Version"],
                         Start_Wealth=start["Pre_Trade_Wealth"], End_Wealth=ending_wealth,
                         Post_Trade_Wealth=start["Post_Trade_Wealth"], Shares=start["Shares"],
                         Pre_Trade_Shares=start["Pre_Trade_Shares"], Pre_Trade_Cash=start["Pre_Trade_Cash"],
                         Pre_Trade_Weight=start["Pre_Trade_Weight"],
                         Cash=start["Cash"], Receivables=start["Receivables"],
                         Target_Weight=start["Target_Weight"], Realized_Weight=start["Realized_Weight"],
                         Signed_Notional=start["Signed_Notional"],
                         Turnover=start["Turnover"]+end["Turnover"]*end["Terminal_Liquidation"],
                         Price_PnL=end["Price_PnL"], Distribution_PnL=end["Distribution_PnL"],
                         Cash_Interest=end["Cash_Interest"], Transaction_Cost=total_cost,
                         Terminal_Cost=end["Terminal_Cost"], Gross_Return=gross_pnl/start["Pre_Trade_Wealth"],
                         Net_Return=net_return, Reference_Return=reference, Net_Excess_Return=net_return-reference))
    daily = pd.DataFrame(rows)
    daily["Net_Equity"] = (1 + daily.Net_Return).cumprod()
    daily["Drawdown"] = daily.Net_Equity / daily.Net_Equity.cummax().clip(lower=1) - 1
    reconciliation = dict(Config=asdict(config), Execution_Source_SHA256=source_hash,
        Instrument_ID=bars.Instrument_ID.iloc[0],
        Dataset_ID=bars.Dataset_ID.iloc[0] if "Dataset_ID" in bars else None,
        Periods=len(daily), Initial_Capital=config.initial_capital,
        Final_Wealth=states[-1]["Post_Trade_Wealth"], Total_Price_PnL=float(daily.Price_PnL.sum()),
        Total_Distribution_PnL=float(daily.Distribution_PnL.sum()), Total_Cash_Interest=float(daily.Cash_Interest.sum()),
        Total_Trading_Cost=float(daily.Transaction_Cost.sum()), Maximum_Relative_Wealth_Identity_Error=maximum_identity_error,
        Outstanding_Receivables=states[-1]["Receivables"], Approximated_Distribution_Events=approximated_distributions,
        Payment_Timing="local_midnight_on_pay_date_or_ex_open_if_later",
        Intraperiod_Cash_Accrual="simple_proration_by_local_calendar_time_with_interest_credited_at_next_open",
        Terminal_Policy="liquidate_equity_at_final_open" if config.terminal_liquidation else "mark_at_final_open")
    bridge = (config.initial_capital + reconciliation["Total_Price_PnL"] + reconciliation["Total_Distribution_PnL"]
              + reconciliation["Total_Cash_Interest"] - reconciliation["Total_Trading_Cost"])
    reconciliation["Pnl_Bridge_Error"] = bridge - reconciliation["Final_Wealth"]
    if abs(reconciliation["Pnl_Bridge_Error"]) > config.relative_tolerance * max(config.initial_capital, bridge) * len(daily):
        raise ArithmeticError("Whole-run P&L reconciliation failed")
    if source_hash != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise RuntimeError("Execution source changed during simulation")
    return PortfolioResult(daily, pd.DataFrame(states), pd.DataFrame(trades, columns=TRADE_COLUMNS),
                           pd.DataFrame(events, columns=EVENT_COLUMNS), targets, reconciliation)


def portfolio_metrics(result: PortfolioResult) -> dict:
    daily = result.daily
    returns = daily.Net_Return.to_numpy(float)
    excess = daily.Net_Excess_Return.to_numpy(float)
    volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    excess_volatility = float(np.std(excess, ddof=1)) if len(excess) > 1 else 0.0
    downside = float(np.sqrt(np.mean(np.minimum(excess, 0)**2)))
    elapsed_years = (daily.Return_End_Time.iloc[-1] - daily.Execution_Time.iloc[0]).total_seconds() / (365.25*86400)
    final_equity = float(daily.Net_Equity.iloc[-1])
    return dict(Periods=len(daily), Net_Total_Return=final_equity-1,
                CAGR=final_equity**(1/elapsed_years)-1, Annualized_Volatility=volatility*np.sqrt(252),
                Sharpe_Ratio=float(np.mean(excess)/excess_volatility*np.sqrt(252)) if excess_volatility > 1e-15 else None,
                Sortino_Ratio=float(np.mean(excess)/downside*np.sqrt(252)) if downside > 1e-15 else None,
                Maximum_Drawdown=float(daily.Drawdown.min()), Annual_Turnover=float(daily.Turnover.sum()*252/len(daily)),
                Trade_Count=len(result.trades), Average_Exposure=float(daily.Realized_Weight.mean()),
                Total_Cost_Dollars=float(daily.Transaction_Cost.sum()),
                Cost_Equity_Fractions=float(result.trades.Turnover.sum()*result.accounting_reconciliation["Config"]["one_way_cost"]))


def save_portfolio_result(result: PortfolioResult, directory) -> Path:
    """Publish a complete immutable result directory after writing its receipt."""
    directory = Path(directory)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if source_hash != result.accounting_reconciliation["Execution_Source_SHA256"]:
        raise RuntimeError("Result execution source differs from the current source")
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        raise FileExistsError(f"A portfolio result already exists at {directory}")
    with tempfile.TemporaryDirectory(prefix=".pending-", dir=directory.parent) as temporary:
        pending = Path(temporary) / directory.name
        pending.mkdir()
        for name, table in [("portfolio_daily", result.daily), ("portfolio_valuations", result.valuations),
                             ("trades", result.trades), ("cash_events", result.cash_events), ("targets", result.targets)]:
            table.to_parquet(pending / f"{name}.parquet", index=False, engine="pyarrow")
        for name, value in [("accounting_reconciliation", result.accounting_reconciliation),
                             ("metrics", portfolio_metrics(result))]:
            (pending / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        manifest = dict(Schema_Version="1.0.0", Execution_Source_SHA256=source_hash,
                        Files={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(pending.iterdir())})
        (pending / "completion.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        publish_directory(pending, directory, root=directory.parent)
    return directory


def _cumulative_at(times, values, valuation_times):
    """Aggregate event flows inclusively at each recorded valuation time."""
    if len(times) == 0:
        return np.zeros(len(valuation_times))
    frame = pd.DataFrame(dict(Time=pd.to_datetime(times, utc=True), Value=np.asarray(values, dtype=float)))
    grouped = frame.groupby("Time").Value.sum().sort_index().cumsum()
    indices = pd.DatetimeIndex(grouped.index).get_indexer(valuation_times, method="pad")
    totals = np.r_[0.0, grouped.to_numpy(float)]
    return totals[indices+1]


def verify_portfolio_result(directory) -> dict:
    """Verify saved bytes and reconstruct balances from the trade/event logs."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory/"completion.json").read_text(encoding="utf-8"))
    if manifest["Execution_Source_SHA256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise DataValidationError("Saved portfolio execution source differs from the current source")
    required = {"portfolio_daily.parquet", "portfolio_valuations.parquet", "trades.parquet", "cash_events.parquet",
                "targets.parquet", "accounting_reconciliation.json", "metrics.json"}
    if set(manifest["Files"]) != required:
        raise DataValidationError("Portfolio receipt has an incomplete artifact set")
    for name, expected in manifest["Files"].items():
        path = directory/name
        if path.name != name or not path.resolve().is_relative_to(directory):
            raise DataValidationError("Unsafe portfolio receipt path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise DataValidationError(f"Portfolio checksum mismatch for {name}")
    daily = pd.read_parquet(directory/"portfolio_daily.parquet")
    states = pd.read_parquet(directory/"portfolio_valuations.parquet")
    trades = pd.read_parquet(directory/"trades.parquet")
    events = pd.read_parquet(directory/"cash_events.parquet")
    receipt = json.loads((directory/"accounting_reconciliation.json").read_text(encoding="utf-8"))
    times = pd.DatetimeIndex(states.Valuation_Time)
    cash = receipt["Initial_Capital"] + _cumulative_at(events.Event_Time, events.Cash_Delta, times)
    cash -= _cumulative_at(trades.Execution_Time, trades.Signed_Notional+trades.Cost, times)
    shares = (_cumulative_at(trades.Execution_Time, trades.Shares_Traded, times)
              + _cumulative_at(events.Event_Time, events.Shares_Delta, times))
    receivables = _cumulative_at(events.Event_Time, events.Receivable_Delta, times)
    checks = [
        (cash, states.Cash), (shares, states.Shares), (receivables, states.Receivables),
        (shares*states.Price+cash+receivables, states.Post_Trade_Wealth),
        (states.Pre_Trade_Wealth-states.Cost, states.Post_Trade_Wealth),
        ((states.Pre_Trade_Wealth.iloc[1:].to_numpy()-states.Terminal_Cost.iloc[1:].to_numpy())
         /states.Pre_Trade_Wealth.iloc[:-1].to_numpy()-1, daily.Net_Return),
        (np.cumprod(1+daily.Net_Return), daily.Net_Equity),
    ]
    for actual, expected in checks:
        if not np.allclose(actual, expected, rtol=1e-10, atol=1e-12*receipt["Initial_Capital"]):
            raise DataValidationError("Saved portfolio balances fail independent reconciliation")
    price_pnl = (states.Pre_Trade_Shares.iloc[1:].to_numpy()*states.Price.iloc[1:].to_numpy()
                 -states.Shares.iloc[:-1].to_numpy()*states.Price.iloc[:-1].to_numpy()).sum()
    distributions = events.loc[events.Event_Type.eq("distribution_entitlement"), "Amount"].sum()
    interest = events.loc[events.Event_Type.eq("cash_interest"), "Cash_Delta"].sum()
    final_wealth = receipt["Initial_Capital"] + price_pnl + distributions + interest - trades.Cost.sum()
    if not np.isclose(final_wealth, states.Post_Trade_Wealth.iloc[-1], rtol=1e-10):
        raise DataValidationError("Saved portfolio fails the independent P&L bridge")
    return dict(Periods=len(daily), Trades=len(trades), Final_Wealth=float(final_wealth))

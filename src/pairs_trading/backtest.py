"""Simple backtester for one pairs trade."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signals import SignalConfig, build_pair_signals


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 10_000.0
    transaction_cost_bps: float = 5.0
    max_holding_days: int = 20


def _apply_max_holding_period(position: pd.Series, max_holding_days: int) -> pd.Series:
    """Force positions flat after max_holding_days in the same trade."""

    if max_holding_days <= 0:
        return position.copy()

    adjusted = []
    current = 0
    days_in_trade = 0
    for desired in position:
        if current == 0:
            current = int(desired)
            days_in_trade = 1 if current != 0 else 0
        elif desired == 0 or desired != current:
            current = int(desired)
            days_in_trade = 1 if current != 0 else 0
        else:
            days_in_trade += 1
            if days_in_trade > max_holding_days:
                current = 0
                days_in_trade = 0
        adjusted.append(current)
    return pd.Series(adjusted, index=position.index, name=position.name)


def backtest_pair(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    signal_config: SignalConfig | None = None,
    backtest_config: BacktestConfig | None = None,
    hedge_ratio: float | None = None,
) -> pd.DataFrame:
    """Backtest a pair using next-day execution.

    The signal is computed from today's close, then applied to tomorrow's
    return. This avoids pretending we can trade on information before it exists.
    """

    signal_config = signal_config or SignalConfig()
    backtest_config = backtest_config or BacktestConfig()
    signals = build_pair_signals(prices, ticker_a, ticker_b, signal_config, hedge_ratio)

    raw_position = signals["position"].astype(int)
    capped_position = _apply_max_holding_period(raw_position, backtest_config.max_holding_days)
    tradable_position = capped_position.shift(1).fillna(0)

    returns_a = signals["price_a"].pct_change().fillna(0)
    returns_b = signals["price_b"].pct_change().fillna(0)
    pair_return_before_cost = tradable_position * (returns_a - signals["hedge_ratio"] * returns_b)

    position_change = tradable_position.diff().abs().fillna(tradable_position.abs())
    cost_rate = backtest_config.transaction_cost_bps / 10_000
    transaction_cost = position_change * cost_rate * (1 + abs(float(signals["hedge_ratio"].iloc[-1])))

    strategy_return = pair_return_before_cost - transaction_cost
    equity = backtest_config.initial_capital * (1 + strategy_return).cumprod()

    result = signals.copy()
    result["raw_position"] = raw_position
    result["position"] = capped_position
    result["tradable_position"] = tradable_position
    result["return_a"] = returns_a
    result["return_b"] = returns_b
    result["gross_strategy_return"] = pair_return_before_cost
    result["transaction_cost"] = transaction_cost
    result["strategy_return"] = strategy_return.replace([np.inf, -np.inf], np.nan).fillna(0)
    result["equity"] = equity
    return result

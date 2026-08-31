"""Performance metrics for the pairs trading backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_metrics(backtest: pd.DataFrame, periods_per_year: int = 252) -> dict[str, float]:
    """Calculate common strategy metrics from a backtest result."""

    returns = backtest["strategy_return"].fillna(0)
    equity = backtest["equity"].ffill()
    if len(returns) == 0:
        raise ValueError("Backtest result is empty.")

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    annualized_return = float((1 + total_return) ** (periods_per_year / max(len(returns), 1)) - 1)
    annualized_volatility = float(returns.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe_ratio = float(annualized_return / annualized_volatility) if annualized_volatility else 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = float(drawdown.min())

    position = backtest["tradable_position"]
    entries = int(((position != 0) & (position.shift(1).fillna(0) == 0)).sum())
    exits = int(((position == 0) & (position.shift(1).fillna(0) != 0)).sum())
    turnover = float(position.diff().abs().fillna(position.abs()).sum())

    trade_returns = _trade_returns(backtest)
    win_rate = float((trade_returns > 0).mean()) if len(trade_returns) else 0.0
    average_trade_return = float(trade_returns.mean()) if len(trade_returns) else 0.0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "entries": entries,
        "exits": exits,
        "turnover": turnover,
        "win_rate": win_rate,
        "average_trade_return": average_trade_return,
    }


def build_trade_log(backtest: pd.DataFrame) -> pd.DataFrame:
    """Create one row per completed or open trade."""

    position = backtest["tradable_position"].fillna(0).astype(int)
    rows = []
    active = None

    for date, pos in position.items():
        if active is None and pos != 0:
            active = {
                "entry_date": date,
                "side": "long_spread" if pos > 0 else "short_spread",
                "entry_equity": float(backtest.loc[date, "equity"]),
                "returns": [],
            }
        elif active is not None and pos == 0:
            exit_equity = float(backtest.loc[date, "equity"])
            rows.append(
                {
                    "entry_date": active["entry_date"],
                    "exit_date": date,
                    "side": active["side"],
                    "holding_days": len(active["returns"]),
                    "trade_return": float(np.prod([1 + r for r in active["returns"]]) - 1),
                    "entry_equity": active["entry_equity"],
                    "exit_equity": exit_equity,
                    "status": "closed",
                }
            )
            active = None

        if active is not None:
            active["returns"].append(float(backtest.loc[date, "strategy_return"]))

    if active is not None:
        last_date = backtest.index[-1]
        rows.append(
            {
                "entry_date": active["entry_date"],
                "exit_date": last_date,
                "side": active["side"],
                "holding_days": len(active["returns"]),
                "trade_return": float(np.prod([1 + r for r in active["returns"]]) - 1),
                "entry_equity": active["entry_equity"],
                "exit_equity": float(backtest["equity"].iloc[-1]),
                "status": "open_at_end",
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "entry_date",
            "exit_date",
            "side",
            "holding_days",
            "trade_return",
            "entry_equity",
            "exit_equity",
            "status",
        ],
    )


def _trade_returns(backtest: pd.DataFrame) -> pd.Series:
    """Group daily returns into completed trade returns."""

    position = backtest["tradable_position"].fillna(0)
    returns = backtest["strategy_return"].fillna(0)

    trade_values = []
    active_returns = []
    previous_position = 0

    for pos, ret in zip(position, returns, strict=True):
        if previous_position == 0 and pos != 0:
            active_returns = [ret]
        elif previous_position != 0 and pos == previous_position:
            active_returns.append(ret)
        elif previous_position != 0 and pos == 0:
            if active_returns:
                trade_values.append(float(np.prod([1 + r for r in active_returns]) - 1))
            active_returns = []
        elif previous_position != 0 and pos != previous_position:
            if active_returns:
                trade_values.append(float(np.prod([1 + r for r in active_returns]) - 1))
            active_returns = [ret]
        previous_position = pos

    if active_returns:
        trade_values.append(float(np.prod([1 + r for r in active_returns]) - 1))

    return pd.Series(trade_values, dtype=float)

"""Parameter sensitivity and optimization helpers."""

from __future__ import annotations

from dataclasses import asdict
from itertools import product

import pandas as pd

from .backtest import BacktestConfig, backtest_pair
from .metrics import calculate_metrics
from .signals import SignalConfig


DEFAULT_LOOKBACKS = [40, 60, 90]
DEFAULT_ENTRY_ZS = [1.5, 2.0, 2.5]
DEFAULT_EXIT_ZS = [0.0, 0.25, 0.5]


def sensitivity_analysis(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    hedge_ratio: float,
    backtest_config: BacktestConfig,
    lookbacks: list[int] | None = None,
    entry_zs: list[float] | None = None,
    exit_zs: list[float] | None = None,
) -> pd.DataFrame:
    """Backtest a parameter grid and return metrics for every valid combo."""

    rows = []
    lookbacks = lookbacks or DEFAULT_LOOKBACKS
    entry_zs = entry_zs or DEFAULT_ENTRY_ZS
    exit_zs = exit_zs or DEFAULT_EXIT_ZS

    for lookback, entry_z, exit_z in product(lookbacks, entry_zs, exit_zs):
        if exit_z >= entry_z:
            continue
        signal_config = SignalConfig(lookback=lookback, entry_z=entry_z, exit_z=exit_z)
        bt = backtest_pair(
            prices[[ticker_a, ticker_b]],
            ticker_a,
            ticker_b,
            signal_config=signal_config,
            backtest_config=backtest_config,
            hedge_ratio=hedge_ratio,
        )
        metrics = calculate_metrics(bt)
        rows.append(
            {
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                **asdict(signal_config),
                **metrics,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["sharpe_ratio", "total_return"],
        ascending=[False, False],
    ).reset_index(drop=True)

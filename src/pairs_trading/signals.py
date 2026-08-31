"""Signal generation for a mean-reverting pair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .stats import calculate_spread, estimate_hedge_ratio, log_prices


@dataclass(frozen=True)
class SignalConfig:
    lookback: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5


def rolling_zscore(series: pd.Series, lookback: int) -> pd.Series:
    """Convert a series into rolling z-scores."""

    mean = series.rolling(lookback).mean()
    std = series.rolling(lookback).std()
    zscore = (series - mean) / std.replace(0, np.nan)
    zscore.name = "zscore"
    return zscore


def build_pair_signals(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    config: SignalConfig | None = None,
    hedge_ratio: float | None = None,
) -> pd.DataFrame:
    """Build spread, z-score, and trading position for a selected pair.

    Position means:
    +1 = long spread: buy A and short hedge_ratio dollars of B.
    -1 = short spread: short A and buy hedge_ratio dollars of B.
     0 = flat.
    """

    config = config or SignalConfig()
    logs = log_prices(prices[[ticker_a, ticker_b]].dropna())

    intercept = 0.0
    if hedge_ratio is None:
        intercept, hedge_ratio = estimate_hedge_ratio(logs[ticker_a], logs[ticker_b])

    spread = calculate_spread(logs[ticker_a], logs[ticker_b], hedge_ratio, intercept)
    zscore = rolling_zscore(spread, config.lookback)

    position = []
    current = 0
    for z in zscore:
        if np.isnan(z):
            position.append(0)
            continue
        if current == 0:
            if z > config.entry_z:
                current = -1
            elif z < -config.entry_z:
                current = 1
        elif abs(z) < config.exit_z:
            current = 0
        position.append(current)

    output = pd.DataFrame(
        {
            "price_a": prices.loc[spread.index, ticker_a],
            "price_b": prices.loc[spread.index, ticker_b],
            "spread": spread,
            "zscore": zscore,
            "position": position,
            "hedge_ratio": hedge_ratio,
        },
        index=spread.index,
    )
    output.index.name = "date"
    return output

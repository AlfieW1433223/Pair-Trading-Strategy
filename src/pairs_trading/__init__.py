"""Beginner-friendly pairs trading research toolkit."""

from .backtest import BacktestConfig, backtest_pair
from .data import generate_demo_prices
from .metrics import calculate_metrics
from .signals import SignalConfig, build_pair_signals
from .stats import find_cointegrated_pairs

__all__ = [
    "BacktestConfig",
    "SignalConfig",
    "backtest_pair",
    "build_pair_signals",
    "calculate_metrics",
    "find_cointegrated_pairs",
    "generate_demo_prices",
]

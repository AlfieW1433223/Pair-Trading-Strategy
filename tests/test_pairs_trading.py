import unittest

import numpy as np

from src.pairs_trading.backtest import BacktestConfig, backtest_pair
from src.pairs_trading.data import generate_demo_prices
from src.pairs_trading.metrics import calculate_metrics
from src.pairs_trading.pipeline import train_test_split_by_date
from src.pairs_trading.signals import SignalConfig, build_pair_signals
from src.pairs_trading.stats import estimate_hedge_ratio, find_cointegrated_pairs


class PairsTradingTests(unittest.TestCase):
    def test_demo_data_has_expected_shape(self):
        prices = generate_demo_prices(rows=250)
        self.assertEqual(prices.shape, (250, 4))
        self.assertTrue((prices > 0).all().all())

    def test_hedge_ratio_estimation(self):
        prices = generate_demo_prices(rows=300)
        intercept, beta = estimate_hedge_ratio(np.log(prices["ALPHA"]), np.log(prices["BETA"]))
        self.assertTrue(np.isfinite(intercept))
        self.assertTrue(0.5 < beta < 1.5)

    def test_cointegrated_pair_search_finds_demo_pair(self):
        prices = generate_demo_prices(rows=600)
        pairs = find_cointegrated_pairs(prices, p_value_threshold=0.10, max_pairs=5)
        found = {tuple(row) for row in pairs[["ticker_a", "ticker_b"]].to_numpy()}
        self.assertIn(("ALPHA", "BETA"), found)

    def test_signals_and_backtest_outputs(self):
        prices = generate_demo_prices(rows=400)
        train, test = train_test_split_by_date(prices)
        self.assertGreater(len(train), len(test))

        signals = build_pair_signals(
            prices[["ALPHA", "BETA"]],
            "ALPHA",
            "BETA",
            SignalConfig(lookback=40, entry_z=1.5, exit_z=0.5),
        )
        self.assertIn("zscore", signals.columns)
        self.assertIn("position", signals.columns)

        backtest = backtest_pair(
            prices[["ALPHA", "BETA"]],
            "ALPHA",
            "BETA",
            SignalConfig(lookback=40, entry_z=1.5, exit_z=0.5),
            BacktestConfig(transaction_cost_bps=5, max_holding_days=20),
        )
        metrics = calculate_metrics(backtest)
        self.assertIn("sharpe_ratio", metrics)
        self.assertTrue(np.isfinite(metrics["total_return"]))


if __name__ == "__main__":
    unittest.main()

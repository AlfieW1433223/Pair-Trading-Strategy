"""End-to-end project pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .backtest import BacktestConfig, backtest_pair
from .data import generate_demo_prices, load_prices_from_csv, save_prices_to_csv
from .metrics import build_trade_log, calculate_metrics
from .optimization import sensitivity_analysis
from .signals import SignalConfig
from .stats import find_cointegrated_pairs, test_all_pairs


def train_test_split_by_date(prices: pd.DataFrame, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split prices into in-sample and out-of-sample periods."""

    if not 0.1 < train_ratio < 0.9:
        raise ValueError("train_ratio should be between 0.1 and 0.9.")
    split = int(len(prices) * train_ratio)
    return prices.iloc[:split].copy(), prices.iloc[split:].copy()


def run_research_pipeline(
    output_dir: str | Path,
    price_csv: str | Path | None = None,
    p_value_threshold: float = 0.05,
    max_pairs: int = 3,
    train_ratio: float = 0.7,
    signal_config: SignalConfig | None = None,
    backtest_config: BacktestConfig | None = None,
    optimize_parameters: bool = True,
) -> dict[str, Path]:
    """Run the full project and save CSV outputs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if price_csv is None:
        prices = generate_demo_prices()
        save_prices_to_csv(prices, output_dir / "demo_prices.csv")
    else:
        prices = load_prices_from_csv(price_csv)
        save_prices_to_csv(prices, output_dir / "input_prices_reference.csv")

    train_prices, test_prices = train_test_split_by_date(prices, train_ratio)
    all_pair_tests = test_all_pairs(train_prices)
    all_pair_tests.to_csv(output_dir / "cointegration_tests_all_pairs.csv", index=False)

    pairs = find_cointegrated_pairs(train_prices, p_value_threshold, max_pairs=max_pairs)
    pairs.to_csv(output_dir / "cointegrated_pairs.csv", index=False)

    metrics_rows = []
    comparison_rows = []
    sensitivity_rows = []
    portfolio_returns = []
    saved_paths = {
        "prices": output_dir / ("demo_prices.csv" if price_csv is None else "input_prices_reference.csv"),
        "all_pair_tests": output_dir / "cointegration_tests_all_pairs.csv",
        "pairs": output_dir / "cointegrated_pairs.csv",
        "metrics": output_dir / "performance_metrics.csv",
        "in_sample_out_of_sample": output_dir / "in_sample_out_of_sample.csv",
        "sensitivity": output_dir / "sensitivity_analysis.csv",
        "portfolio": output_dir / "portfolio_equity.csv",
    }

    if pairs.empty:
        pd.DataFrame(metrics_rows).to_csv(saved_paths["metrics"], index=False)
        pd.DataFrame(comparison_rows).to_csv(saved_paths["in_sample_out_of_sample"], index=False)
        pd.DataFrame(sensitivity_rows).to_csv(saved_paths["sensitivity"], index=False)
        pd.DataFrame().to_csv(saved_paths["portfolio"], index=False)
        return saved_paths

    for _, pair in pairs.iterrows():
        ticker_a = str(pair["ticker_a"])
        ticker_b = str(pair["ticker_b"])
        hedge_ratio = float(pair["hedge_ratio"])
        pair_name = f"{ticker_a}_{ticker_b}"

        sensitivity = sensitivity_analysis(
            train_prices[[ticker_a, ticker_b]],
            ticker_a,
            ticker_b,
            hedge_ratio,
            backtest_config or BacktestConfig(),
        )
        sensitivity["cointegration_p_value"] = float(pair["p_value"])
        sensitivity_rows.append(sensitivity)

        if optimize_parameters and not sensitivity.empty:
            best = sensitivity.iloc[0]
            chosen_signal_config = SignalConfig(
                lookback=int(best["lookback"]),
                entry_z=float(best["entry_z"]),
                exit_z=float(best["exit_z"]),
            )
        else:
            chosen_signal_config = signal_config or SignalConfig()

        train_bt = backtest_pair(
            train_prices[[ticker_a, ticker_b]],
            ticker_a,
            ticker_b,
            signal_config=chosen_signal_config,
            backtest_config=backtest_config,
            hedge_ratio=hedge_ratio,
        )

        test_with_warmup = pd.concat([train_prices.tail(80), test_prices])
        bt = backtest_pair(
            test_with_warmup[[ticker_a, ticker_b]],
            ticker_a,
            ticker_b,
            signal_config=chosen_signal_config,
            backtest_config=backtest_config,
            hedge_ratio=hedge_ratio,
        )
        bt = bt.loc[test_prices.index.intersection(bt.index)]

        backtest_path = output_dir / f"backtest_{pair_name}.csv"
        bt.to_csv(backtest_path)
        saved_paths[f"backtest_{pair_name}"] = backtest_path

        trade_log_path = output_dir / f"trade_log_{pair_name}.csv"
        build_trade_log(bt).to_csv(trade_log_path, index=False)
        saved_paths[f"trade_log_{pair_name}"] = trade_log_path

        train_metrics = calculate_metrics(train_bt)
        test_metrics = calculate_metrics(bt)
        for period_name, period_metrics in [("in_sample", train_metrics), ("out_of_sample", test_metrics)]:
            comparison_rows.append(
                {
                    "ticker_a": ticker_a,
                    "ticker_b": ticker_b,
                    "period": period_name,
                    "lookback": chosen_signal_config.lookback,
                    "entry_z": chosen_signal_config.entry_z,
                    "exit_z": chosen_signal_config.exit_z,
                    **period_metrics,
                }
            )

        metrics = calculate_metrics(bt)
        metrics.update(
            {
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "hedge_ratio": hedge_ratio,
                "cointegration_p_value": float(pair["p_value"]),
                "lookback": chosen_signal_config.lookback,
                "entry_z": chosen_signal_config.entry_z,
                "exit_z": chosen_signal_config.exit_z,
            }
        )
        metrics_rows.append(metrics)
        portfolio_returns.append(bt["strategy_return"].rename(pair_name))

    pd.DataFrame(metrics_rows).to_csv(saved_paths["metrics"], index=False)
    pd.concat(sensitivity_rows, ignore_index=True).to_csv(saved_paths["sensitivity"], index=False)
    pd.DataFrame(comparison_rows).to_csv(saved_paths["in_sample_out_of_sample"], index=False)

    if portfolio_returns:
        returns = pd.concat(portfolio_returns, axis=1).fillna(0)
        portfolio = pd.DataFrame(index=returns.index)
        portfolio["portfolio_return"] = returns.mean(axis=1)
        portfolio["portfolio_equity"] = 10_000 * (1 + portfolio["portfolio_return"]).cumprod()
        portfolio.to_csv(saved_paths["portfolio"])
    else:
        pd.DataFrame().to_csv(saved_paths["portfolio"], index=False)

    return saved_paths

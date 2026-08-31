"""Command-line entrypoint for the pairs trading project."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pairs_trading.backtest import BacktestConfig
from src.pairs_trading.data import DataConfig, download_prices, download_sp500_prices, save_prices_to_csv
from src.pairs_trading.pipeline import run_research_pipeline
from src.pairs_trading.signals import SignalConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple pairs trading research project.")
    parser.add_argument("--price-csv", default=None, help="Optional CSV with dates as rows and tickers as columns.")
    parser.add_argument("--sp500", action="store_true", help="Download current S&P 500 constituent prices.")
    parser.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated Yahoo Finance tickers to download, for example KO,PEP,V,MA.",
    )
    parser.add_argument("--start", default="2018-01-01", help="Download start date.")
    parser.add_argument("--end", default="2026-08-31", help="Download end date.")
    parser.add_argument("--max-tickers", type=int, default=50, help="Maximum S&P 500 tickers to download.")
    parser.add_argument("--output-dir", default="outputs", help="Folder for generated CSV outputs.")
    parser.add_argument("--p-value", type=float, default=0.05, help="Cointegration p-value threshold.")
    parser.add_argument("--max-pairs", type=int, default=3, help="Maximum number of pairs to backtest.")
    parser.add_argument("--lookback", type=int, default=60, help="Rolling z-score lookback window.")
    parser.add_argument("--entry-z", type=float, default=2.0, help="Entry z-score threshold.")
    parser.add_argument("--exit-z", type=float, default=0.5, help="Exit z-score threshold.")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Transaction cost in basis points.")
    parser.add_argument("--max-holding-days", type=int, default=20, help="Maximum days to hold one trade.")
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Use the fixed lookback/entry/exit parameters instead of choosing the best in-sample grid result.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    price_csv = args.price_csv
    output_dir = Path(args.output_dir)
    data_config = DataConfig(start=args.start, end=args.end)

    if args.sp500 and args.tickers:
        raise SystemExit("Use either --sp500 or --tickers, not both.")
    if args.sp500 and args.price_csv:
        raise SystemExit("Use either --sp500 or --price-csv, not both.")
    if args.tickers and args.price_csv:
        raise SystemExit("Use either --tickers or --price-csv, not both.")

    if args.sp500:
        prices = download_sp500_prices(max_tickers=args.max_tickers, config=data_config)
        price_csv = output_dir / "real_sp500_prices.csv"
        save_prices_to_csv(prices, price_csv)
    elif args.tickers:
        tickers = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
        prices = download_prices(tickers, config=data_config)
        price_csv = output_dir / "real_prices.csv"
        save_prices_to_csv(prices, price_csv)

    paths = run_research_pipeline(
        output_dir=output_dir,
        price_csv=price_csv,
        p_value_threshold=args.p_value,
        max_pairs=args.max_pairs,
        signal_config=SignalConfig(
            lookback=args.lookback,
            entry_z=args.entry_z,
            exit_z=args.exit_z,
        ),
        backtest_config=BacktestConfig(
            transaction_cost_bps=args.cost_bps,
            max_holding_days=args.max_holding_days,
        ),
        optimize_parameters=not args.no_optimize,
    )

    print("Project finished. Files created:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()

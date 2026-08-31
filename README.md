# Simple Pairs Trading Strategy

This project implements the strategy from `4. Pairs Trading Strategy.pdf` in a beginner-friendly way.

It does four things:

1. Finds stock pairs that may be cointegrated.
2. Builds a spread and z-score trading signal.
3. Backtests long/short mean-reversion trades.
4. Saves tables you can inspect: candidate pairs, trade history, and performance metrics.

This is for learning and research. It is not financial advice and it is not a guarantee of profit.

## The Trading Idea

Pairs trading looks for two assets with a stable relationship. When the relationship stretches too far, the strategy bets that it will move back toward normal.

For a pair `A` and `B`, the project estimates:

```text
log(A) = intercept + hedge_ratio * log(B)
```

Then it calculates:

```text
spread = log(A) - intercept - hedge_ratio * log(B)
zscore = (spread - rolling_average_spread) / rolling_spread_std
```

Trading rules:

```text
If zscore > entry threshold:
    short A and buy B

If zscore < -entry threshold:
    buy A and short B

If abs(zscore) < exit threshold:
    close the trade
```

## Project Files

```text
run_project.py                 Main command-line runner
src/pairs_trading/data.py      Data loading and demo data generation
src/pairs_trading/stats.py     Hedge ratio and cointegration logic
src/pairs_trading/signals.py   Spread, z-score, and position rules
src/pairs_trading/backtest.py  Backtesting engine
src/pairs_trading/metrics.py   Performance metrics
tests/test_pairs_trading.py    Basic automated tests
outputs/                       Generated research outputs
```

## Quick Start

Run the offline synthetic-data demo:

```bash
python3 run_project.py
```

This creates:

```text
outputs/demo_prices.csv
outputs/cointegration_tests_all_pairs.csv
outputs/cointegrated_pairs.csv
outputs/backtest_ALPHA_BETA.csv
outputs/trade_log_ALPHA_BETA.csv
outputs/performance_metrics.csv
outputs/in_sample_out_of_sample.csv
outputs/sensitivity_analysis.csv
outputs/portfolio_equity.csv
```

Run on real S&P 500 constituent data:

```bash
python3 RunCodes.py --sp500 --max-tickers 50 --output-dir outputs_real --start 2018-01-01 --end 2026-08-31 --max-pairs 5 --cost-bps 10
```

`RunCodes.py` is included as a compatibility wrapper for reviewers. It calls the same code as `run_project.py`.

Run tests:

```bash
python3 -m unittest discover tests
```

## Using Real Stock Data

Create a CSV with:

- First column: dates
- Other columns: ticker prices

Example:

```text
date,KO,PEP,V,MA
2020-01-02,48.1,126.7,188.7,300.5
2020-01-03,47.8,125.9,186.9,297.2
```

Then run:

```bash
python3 run_project.py --price-csv your_prices.csv
```

You can also download a specific real-stock universe:

```bash
python3 RunCodes.py --tickers KO,PEP,V,MA,XOM,CVX --output-dir outputs_real
```

## Important Parameters

```text
--p-value              Cointegration threshold. Lower is stricter.
--sp500                Download current S&P 500 constituent prices.
--tickers              Download a comma-separated custom ticker list.
--start                Download start date.
--end                  Download end date.
--max-tickers          Cap the S&P 500 universe for a manageable run.
--lookback             Rolling window for z-score calculation.
--entry-z              How extreme the spread must be before entering.
--exit-z               How close to normal before exiting.
--cost-bps             Transaction cost in basis points.
--max-holding-days     Force-exit trades after this many days.
--no-optimize          Use your fixed parameters instead of grid-searching in-sample.
```

Example:

```bash
python3 run_project.py --entry-z 1.5 --exit-z 0.25 --cost-bps 10
```

By default, the final out-of-sample backtest uses the best in-sample settings from the sensitivity grid. Add `--no-optimize` if you want to force the exact parameter values you passed on the command line.

## How to Read the Results

`cointegrated_pairs.csv` shows candidate pairs discovered in the training period.

`cointegration_tests_all_pairs.csv` shows every pair tested, including pairs rejected by the threshold.

`backtest_*.csv` shows daily values:

- `spread`
- `zscore`
- `position`
- `strategy_return`
- `transaction_cost`
- `equity`

`trade_log_*.csv` shows one row per trade.

`performance_metrics.csv` summarizes:

- total return
- annualized return
- volatility
- Sharpe ratio
- max drawdown
- entries and exits
- win rate
- average trade return

`sensitivity_analysis.csv` shows how performance changes across lookback, entry, and exit settings.

`in_sample_out_of_sample.csv` compares the training-period result with the unseen testing-period result.

`portfolio_equity.csv` combines selected pair returns into a simple equal-weight portfolio.

## Real Data Sample Run

I ran the project on a 50-stock sample of current S&P 500 constituents using Yahoo Finance adjusted close prices from 2018-01-01 through 2026-08-31. The pipeline selected pairs in the in-sample period, optimized parameters in-sample, then evaluated them out-of-sample with 10 bps transaction costs.

Example out-of-sample results from `outputs_real/performance_metrics.csv`:

```text
Pair       Total Return    Sharpe    Max Drawdown
AMP/APH       -18.65%      -0.23       -65.31%
ABT/AWK       -35.77%      -0.72       -38.81%
ATO/MMM       -25.53%      -0.53       -45.78%
ACN/GOOGL     -30.16%      -0.40       -68.78%
ACN/GOOG      -67.08%      -1.02       -75.51%
```

These real-data results are intentionally shown separately from the synthetic demo. They suggest the simple version of the strategy does not currently survive out-of-sample testing on this sample, which is an important and realistic quant research finding.

## Notes About Results

The default `python3 run_project.py` command uses synthetic demo data. Those numbers are only meant to prove that the pipeline works and to make the trading logic easy to inspect.

For resume/GitHub credibility, use the real-data command and discuss the results from `outputs_real/`, not the synthetic ALPHA/BETA output. Very high Sharpe ratios on synthetic data should not be presented as real strategy performance.

## What to Improve Next

Good next upgrades:

1. Add charts for spread, z-score, and equity.
2. Add stricter multiple-testing controls.
3. Add survivorship-bias-aware historical S&P 500 membership.
4. Add short-borrow costs and slippage assumptions.
5. Add paper-trading mode before any live trading.

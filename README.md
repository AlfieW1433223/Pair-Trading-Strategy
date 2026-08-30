# Pairs Trading Strategy Notes

## Project Summary

This project implements a simple statistical arbitrage strategy based on pairs trading and cointegration. The main idea is to identify two assets whose prices have historically maintained a stable long-run relationship, then trade temporary deviations from that relationship.

If the spread between the two assets becomes unusually high, the strategy shorts the relatively expensive asset and buys the relatively cheap asset. If the spread becomes unusually low, it does the opposite. The trade is closed when the spread mean-reverts toward normal.

This is a research/backtesting project, not a live trading system or financial advice.

## Strategy Logic

For a pair of assets `A` and `B`, the strategy works with log prices:

```text
log(A) = intercept + hedge_ratio * log(B)
```

The hedge ratio is estimated using ordinary least squares regression. After estimating the relationship, the spread is:

```text
spread = log(A) - intercept - hedge_ratio * log(B)
```

The spread is then normalized into a rolling z-score:

```text
zscore = (spread - rolling_mean(spread)) / rolling_std(spread)
```

The z-score measures how unusual the current spread is compared with its recent history.

## Trading Rules

The strategy uses threshold-based mean-reversion rules:

```text
If zscore > entry_z:
    Short A and buy B

If zscore < -entry_z:
    Buy A and short B

If abs(zscore) < exit_z:
    Close the trade
```

The project also includes a maximum holding period, so a trade can be forced closed if it does not mean-revert quickly enough.

## Cointegration

Correlation is not enough for pairs trading. Two stocks can be highly correlated but still drift apart over time.

Cointegration is stronger. It means two price series may each move around individually, but a combination of them remains relatively stable. In this project, candidate pairs are selected by testing whether the residual spread appears mean-reverting.

The project includes an Engle-Granger-style cointegration workflow:

1. Estimate the hedge ratio between two log-price series.
2. Construct the residual spread.
3. Test whether the spread appears stationary/mean-reverting.
4. Keep pairs whose p-values pass a selected threshold.

## Backtesting

The backtest simulates how the strategy would have performed historically. It includes:

- Next-period execution logic to reduce lookahead bias.
- Transaction costs in basis points.
- Maximum holding period.
- Daily strategy returns.
- Cumulative equity curve.
- Trade-by-trade logs.

The project separates the data into in-sample and out-of-sample periods:

```text
In-sample:
    Used to discover pairs and tune parameters.

Out-of-sample:
    Used to test whether the strategy works on unseen data.
```

This is important because a strategy that only works on the data used to design it may be overfit.

## Parameters Tested

The project supports parameter sensitivity analysis across:

- Cointegration p-value threshold.
- Rolling lookback window.
- Entry z-score threshold.
- Exit z-score threshold.
- Transaction cost assumption.
- Maximum holding period.

The default pipeline chooses the best in-sample parameter setting from a small grid, then evaluates the selected configuration out of sample.

## Performance Metrics

The project reports:

- Total return.
- Annualized return.
- Annualized volatility.
- Sharpe ratio.
- Maximum drawdown.
- Number of entries and exits.
- Turnover.
- Win rate.
- Average trade return.

It also creates a simple equal-weight portfolio equity curve across selected pairs.

## Files Produced

The main command is:

```bash
python3 run_project.py
```

The generated outputs include:

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

The project also includes unit tests:

```bash
python3 -m unittest discover tests
```

## Current Demo Results

The current demo uses synthetic/offline data, so the results are mainly for validating the workflow and learning the mechanics. The synthetic data intentionally contains at least one pair with a stable relationship.

Example selected pair:

```text
Pair: ALPHA / BETA
Hedge ratio: about 1.0263
Cointegration p-value: about 1.29e-16
Selected parameters: lookback 60, entry z-score 1.5, exit z-score 0.25
```

Out-of-sample demo metrics for `ALPHA / BETA`:

```text
Total return: about 135.1%
Annualized return: about 122.1%
Annualized volatility: about 34.4%
Sharpe ratio: about 3.55
Max drawdown: about -9.8%
Win rate: about 93.3%
```

These numbers should not be interpreted as realistic live trading results because the data is generated. The next step would be replacing demo data with real stock data and checking whether results survive transaction costs, slippage, liquidity constraints, and true out-of-sample testing.

## Possible Next Improvements

Useful extensions would include:

- Real S&P 500 historical price data.
- Adjusted close prices from `yfinance` or another data vendor.
- Charts for spread, z-score, trades, and equity.
- Stricter multiple-testing controls.
- More realistic slippage and short-borrow costs.
- Portfolio risk limits.
- Paper-trading mode before any live trading.

## Key Takeaway

The project demonstrates a complete beginner quant workflow:

```text
Hypothesis
-> data
-> cointegration test
-> signal construction
-> backtest
-> transaction costs
-> parameter sensitivity
-> out-of-sample validation
-> performance analysis
```

The main lesson is not that this exact strategy is guaranteed to make money. The value is learning how to structure, test, and evaluate a trading idea without accidentally overfitting or using future information.

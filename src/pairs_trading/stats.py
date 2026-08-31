"""Statistical tools used by the pairs trading strategy."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import erfc, sqrt

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CointegrationResult:
    ticker_a: str
    ticker_b: str
    hedge_ratio: float
    p_value: float
    test_stat: float
    method: str


def log_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert prices to log prices."""

    if (prices <= 0).any().any():
        raise ValueError("Prices must be positive before taking logs.")
    return np.log(prices)


def estimate_hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """Estimate y = intercept + hedge_ratio * x using ordinary least squares."""

    aligned = pd.concat([y, x], axis=1).dropna()
    if len(aligned) < 3:
        raise ValueError("Need at least three observations for regression.")

    y_values = aligned.iloc[:, 0].to_numpy()
    x_values = aligned.iloc[:, 1].to_numpy()
    design = np.column_stack([np.ones(len(x_values)), x_values])
    intercept, hedge_ratio = np.linalg.lstsq(design, y_values, rcond=None)[0]
    return float(intercept), float(hedge_ratio)


def calculate_spread(y: pd.Series, x: pd.Series, hedge_ratio: float, intercept: float = 0.0) -> pd.Series:
    """Calculate residual spread from y - intercept - hedge_ratio * x."""

    aligned = pd.concat([y, x], axis=1).dropna()
    spread = aligned.iloc[:, 0] - intercept - hedge_ratio * aligned.iloc[:, 1]
    spread.name = "spread"
    return spread


def approximate_adf_test(series: pd.Series) -> tuple[float, float]:
    """Small fallback ADF-style test for environments without statsmodels.

    This estimates delta(series) = alpha + rho * lag(series). The p-value is a
    rough normal approximation, not a MacKinnon cointegration p-value. It is
    good enough for learning and automated tests, but use statsmodels for real
    research.
    """

    values = pd.Series(series).dropna().to_numpy()
    if len(values) < 20:
        return 0.0, 1.0

    delta = np.diff(values)
    lagged = values[:-1]
    design = np.column_stack([np.ones(len(lagged)), lagged])
    coeffs = np.linalg.lstsq(design, delta, rcond=None)[0]
    residuals = delta - design @ coeffs
    degrees = len(delta) - design.shape[1]
    if degrees <= 0:
        return 0.0, 1.0

    sigma2 = float((residuals @ residuals) / degrees)
    xtx_inv = np.linalg.pinv(design.T @ design)
    standard_error = sqrt(max(sigma2 * xtx_inv[1, 1], 1e-12))
    test_stat = float(coeffs[1] / standard_error)

    # One-sided probability of seeing a negative value this extreme.
    p_value = 0.5 * erfc(abs(test_stat) / sqrt(2))
    return test_stat, float(min(max(p_value, 0.0), 1.0))


def engle_granger_test(y: pd.Series, x: pd.Series) -> tuple[float, float, float, str]:
    """Return hedge ratio, p-value, test statistic, and method name."""

    try:
        from statsmodels.tsa.stattools import coint

        score, p_value, _ = coint(y.dropna(), x.dropna())
        intercept, hedge_ratio = estimate_hedge_ratio(y, x)
        return hedge_ratio, float(p_value), float(score), "statsmodels.coint"
    except Exception:
        intercept, hedge_ratio = estimate_hedge_ratio(y, x)
        spread = calculate_spread(y, x, hedge_ratio, intercept)
        test_stat, p_value = approximate_adf_test(spread)
        return hedge_ratio, p_value, test_stat, "approx_adf_fallback"


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    p_value_threshold: float = 0.05,
    max_pairs: int | None = None,
) -> pd.DataFrame:
    """Test every pair and return cointegrated candidates sorted by p-value."""

    result = test_all_pairs(prices)
    result = result[result["p_value"] <= p_value_threshold].reset_index(drop=True)
    if max_pairs is not None:
        result = result.head(max_pairs)
    return result


def test_all_pairs(prices: pd.DataFrame) -> pd.DataFrame:
    """Test every pair and return the full p-value table."""

    logs = log_prices(prices)
    rows: list[CointegrationResult] = []

    for ticker_a, ticker_b in combinations(logs.columns, 2):
        hedge_ratio, p_value, test_stat, method = engle_granger_test(logs[ticker_a], logs[ticker_b])
        rows.append(
            CointegrationResult(
                ticker_a=ticker_a,
                ticker_b=ticker_b,
                hedge_ratio=hedge_ratio,
                p_value=p_value,
                test_stat=test_stat,
                method=method,
            )
        )

    result = pd.DataFrame([row.__dict__ for row in rows])
    if result.empty:
        return pd.DataFrame(
            columns=["ticker_a", "ticker_b", "hedge_ratio", "p_value", "test_stat", "method"]
        )

    return result.sort_values(["p_value", "ticker_a", "ticker_b"]).reset_index(drop=True)

"""Data loading helpers for the pairs trading project."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataConfig:
    """Settings used when fetching or creating price data."""

    start: str = "2018-01-01"
    end: str = "2026-08-31"
    min_non_null_ratio: float = 0.95


def generate_demo_prices(rows: int = 900, seed: int = 7) -> pd.DataFrame:
    """Create an offline price dataset with one intentionally cointegrated pair.

    ALPHA and BETA share a stable long-run relationship. GAMMA and DELTA are
    independent random walks and should usually look much less attractive.
    """

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=rows)

    market = np.cumsum(rng.normal(0.00025, 0.01, rows))
    beta_noise = rng.normal(0, 0.015, rows)
    spread = np.zeros(rows)
    for i in range(1, rows):
        spread[i] = 0.92 * spread[i - 1] + rng.normal(0, 0.018)

    alpha_log = np.log(80) + market + 0.55 * spread
    beta_log = np.log(75) + market + beta_noise - 0.45 * spread

    gamma_log = np.log(40) + np.cumsum(rng.normal(0.0004, 0.018, rows))
    delta_log = np.log(120) + np.cumsum(rng.normal(0.0001, 0.02, rows))

    prices = pd.DataFrame(
        {
            "ALPHA": np.exp(alpha_log),
            "BETA": np.exp(beta_log),
            "GAMMA": np.exp(gamma_log),
            "DELTA": np.exp(delta_log),
        },
        index=dates,
    )
    prices.index.name = "date"
    return prices.round(4)


def load_prices_from_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV where rows are dates and columns are tickers."""

    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    return clean_prices(prices)


def save_prices_to_csv(prices: pd.DataFrame, path: str | Path) -> None:
    """Save prices in the format expected by load_prices_from_csv."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path)


def clean_prices(prices: pd.DataFrame, min_non_null_ratio: float = 0.95) -> pd.DataFrame:
    """Remove unusable columns and align all tickers on the same dates."""

    if prices.empty:
        raise ValueError("Price data is empty.")

    prices = prices.sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.loc[:, prices.notna().mean() >= min_non_null_ratio]
    prices = prices.ffill().dropna()
    prices = prices.loc[:, prices.gt(0).all()]

    if prices.shape[1] < 2:
        raise ValueError("Need at least two usable price columns.")
    return prices


def download_prices(
    tickers: list[str],
    config: DataConfig | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance using yfinance.

    This function is optional. The rest of the project can run with a CSV or
    the generated demo data when internet access or yfinance is unavailable.
    """

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. Run `pip install -r requirements.txt` "
            "or use `--demo`."
        ) from exc

    config = config or DataConfig()
    raw = yf.download(
        tickers,
        start=config.start,
        end=config.end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    return clean_prices(prices, config.min_non_null_ratio)


def get_sp500_tickers(max_tickers: int | None = None) -> list[str]:
    """Fetch current S&P 500 tickers from Wikipedia.

    Yahoo Finance uses hyphens for tickers that Wikipedia writes with dots,
    such as BRK.B -> BRK-B.
    """

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "requests is required for --sp500. Run `pip install -r requirements.txt`."
        ) from exc

    response = requests.get(
        url,
        headers={"User-Agent": "pairs-trading-research/1.0"},
        timeout=30,
    )
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    constituents = tables[0]
    tickers = constituents["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    if max_tickers is not None:
        tickers = tickers[:max_tickers]
    return tickers


def download_sp500_prices(
    max_tickers: int | None = 50,
    config: DataConfig | None = None,
) -> pd.DataFrame:
    """Fetch prices for current S&P 500 constituents."""

    tickers = get_sp500_tickers(max_tickers=max_tickers)
    return download_prices(tickers, config=config)

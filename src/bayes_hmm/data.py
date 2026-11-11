"""Real market data (daily adjusted prices, Yahoo Finance) and, for the unit
tests only, a simulator of regime-switching returns with known ground truth."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def simulate_regimes(T: int = 2500, mu=(-0.0015, 0.0008), sigma=(0.020, 0.008),
                     A=((0.97, 0.03), (0.01, 0.99)), seed: int = 0):
    """Daily-return-like series from a K-state Gaussian HMM (used by the tests,
    where known true regimes make correctness checkable; not by the experiments).

    Defaults: state 0 = "bear" (negative drift, high volatility), state 1 =
    "bull" (positive drift, low volatility). Returns (returns, true_states).
    """
    rng = np.random.default_rng(seed)
    mu, sigma, A = np.asarray(mu), np.asarray(sigma), np.asarray(A)
    K = len(mu)
    # start from the stationary distribution of A
    w, v = np.linalg.eig(A.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    pi = pi / pi.sum()
    s = np.empty(T, dtype=int)
    s[0] = rng.choice(K, p=pi)
    for t in range(1, T):
        s[t] = rng.choice(K, p=A[s[t - 1]])
    y = mu[s] + sigma[s] * rng.normal(size=T)
    return y, s


DEFAULT_START = "2000-01-01"
DEFAULT_END = "2026-11-01"          # exclusive; fixed so every run sees the same days


def load_prices(ticker: str = "SPY", start: str = DEFAULT_START, end: str | None = DEFAULT_END,
                csv: str | None = None, cache_dir: str | Path | None = "data"):
    """Daily adjusted closing prices as a pandas Series indexed by date.

    Order of preference: an explicit ``csv`` (Date + "Adj Close" or "Close"
    columns, e.g. Yahoo Finance's Historical Data download); a cached
    ``data/<ticker>.csv`` from an earlier run; otherwise a download from Yahoo
    Finance with yfinance (``pip install -e ".[real-data]"``), which is then
    cached. Adjusted closes include dividends and splits. Later dividends
    rescale the whole adjusted history by a constant, so log returns, and
    every result in this repo, do not change between downloads.
    """
    import pandas as pd

    cache = Path(cache_dir) / f"{ticker}.csv" if cache_dir is not None else None
    path = csv or (cache if cache is not None and cache.exists() else None)
    if path is None:
        import yfinance as yf

        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.index.name = "Date"
        if df.empty:
            raise RuntimeError(f"no data downloaded for {ticker}")
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            df[["Close", "Adj Close"]].to_csv(cache)
        path = cache
        px = df["Adj Close"]
    else:
        df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
        px = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    px = pd.to_numeric(px, errors="coerce").dropna()
    px = px[px.index >= pd.Timestamp(start)]
    if end is not None:
        px = px[px.index < pd.Timestamp(end)]
    if (px <= 0).any() or px.index.duplicated().any():
        raise ValueError("prices must be positive with unique dates")
    return px


def load_returns(ticker: str = "SPY", start: str = DEFAULT_START, end: str | None = DEFAULT_END,
                 csv: str | None = None, cache_dir: str | Path | None = "data"):
    """Daily log returns (numpy array) and their dates (pandas DatetimeIndex)."""
    px = load_prices(ticker, start, end, csv, cache_dir)
    r = np.log(px).diff().dropna()
    return r.to_numpy(), r.index

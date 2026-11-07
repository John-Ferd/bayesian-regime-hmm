"""Synthetic regime-switching returns, and an optional real-data loader."""
from __future__ import annotations

import numpy as np


def simulate_regimes(T: int = 2500, mu=(-0.0015, 0.0008), sigma=(0.020, 0.008),
                     A=((0.97, 0.03), (0.01, 0.99)), seed: int = 0):
    """Daily-return-like series from a K-state Gaussian HMM.

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


def load_returns(ticker: str = "SPY", start: str = "2005-01-01", end: str | None = None, csv: str | None = None):
    """Daily log returns of a price series.

    With ``csv``: a file of daily prices with a Date column and an "Adj Close"
    (preferred, adjusted for dividends and splits) or "Close" column, e.g. the
    download from Yahoo Finance's Historical Data page. Without it: download
    from Yahoo Finance (needs `pip install yfinance` and internet).
    """
    import pandas as pd

    if csv is not None:
        df = pd.read_csv(csv, parse_dates=["Date"]).sort_values("Date").set_index("Date")
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        px = pd.to_numeric(df[col], errors="coerce").dropna()
        px = px[px.index >= pd.Timestamp(start)]
        if end is not None:
            px = px[px.index <= pd.Timestamp(end)]
    else:
        import yfinance as yf

        px = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    r = np.log(px).diff().dropna()
    return r.to_numpy(), r.index

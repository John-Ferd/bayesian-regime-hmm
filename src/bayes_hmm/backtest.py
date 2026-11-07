"""
Causal regime-filtered trading rules and their evaluation.

Every position for day t+1 is decided from returns up to day t only. The
probability of being in the low-volatility ("bull") regime tomorrow is

    P(s_{t+1} = 1 | y_1..t) = sum_k P(s_t = k | y_1..t) * A[k, 1]

computed by forward filtering, which never looks ahead. With a Bayesian fit
this is averaged over posterior draws of (A, mu, sigma), so parameter
uncertainty is carried into the position instead of being ignored.
"""
from __future__ import annotations

import numpy as np


def predictive_bull_prob(y, mu, sigma, A, bull: int = 1) -> np.ndarray:
    """P(s_{t+1} = bull | y_1..t) for t = 0..T-1, averaged over D parameter draws.

    mu, sigma: (D, K); A: (D, K, K). Returns an array of length T whose entry t
    is the forecast for day t+1, made at the close of day t.
    """
    y = np.asarray(y, dtype=float)
    mu, sigma, A = np.atleast_2d(mu), np.atleast_2d(sigma), np.asarray(A)
    if A.ndim == 2:
        A = A[None]
    D, K = mu.shape
    T = len(y)
    logp = -np.log(sigma)[None] - 0.5 * ((y[:, None, None] - mu[None]) / sigma[None]) ** 2   # (T, D, K)
    logp -= logp.max(axis=2, keepdims=True)
    B = np.exp(logp)
    filt = np.full((D, K), 1.0 / K)
    out = np.empty(T)
    for t in range(T):
        prior_t = filt if t == 0 else np.einsum("dk,dkj->dj", filt, A)
        f = prior_t * B[t]
        filt = f / f.sum(axis=1, keepdims=True)
        out[t] = np.einsum("dk,dk->d", filt, A[:, :, bull]).mean()
    return out


def performance(returns, periods: int = 252) -> dict:
    """Annualized Sharpe (zero risk-free rate), total log return, max drawdown."""
    r = np.asarray(returns, dtype=float)
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(periods) if r.std() > 0 else 0.0
    wealth = np.exp(np.cumsum(r))
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    max_dd = float((wealth / peak - 1).min())
    return {"sharpe": float(sharpe), "total_log_return": float(r.sum()), "max_drawdown": max_dd}


def backtest(y, positions, cost_bps: float = 1.0) -> dict:
    """Apply positions (decided the day before) to log returns y.

    positions[t] is the exposure held during day t+1. Strategy log return is
    approximated by position * y (fine for daily returns), minus a cost of
    ``cost_bps`` basis points per unit of turnover.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(positions, dtype=float)
    held = w[:-1]                       # exposure during days 1..T-1
    turnover = np.abs(np.diff(np.concatenate([[0.0], held])))
    strat = held * y[1:] - cost_bps * 1e-4 * turnover
    res = performance(strat)
    res["avg_exposure"] = float(held.mean())
    res["turnover_per_year"] = float(turnover.sum() / len(held) * 252)
    return res

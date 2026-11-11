"""
Causal regime forecasts, exposure rules, and their evaluation.

Every position for day t+1 is decided from returns up to day t only. The
regime forecast for tomorrow is

    P(s_{t+1} = j | y_1..t) = sum_k P(s_t = k | y_1..t) * A[k, j]

computed by forward filtering, which never looks ahead. With a Bayesian fit
this is averaged over posterior draws of (A, mu, sigma), so parameter
uncertainty is carried into the position instead of being ignored.

States are ordered by volatility (state 0 = most turbulent, see gibbs.py), so
for any K the "calmest" state is K - 1 and the "turbulent" state is 0.

Exposure rules (all in [0, 1]):
    hard            1 if P(turbulent tomorrow) < 0.5 else 0
    soft            1 - P(turbulent tomorrow)
    soft_confident  soft * (1 - normalized entropy of tomorrow's regime forecast):
                    hold less when the model cannot tell the regimes apart
                    (an idea from an earlier version of this project, now
                    evaluated out of sample)
"""
from __future__ import annotations

import numpy as np

PERIODS = 252


def predictive_state_prob(y, mu, sigma, A) -> np.ndarray:
    """P(s_{t+1} = k | y_1..t) for t = 0..T-1, averaged over D parameter draws.

    mu, sigma: (D, K) or (K,); A: (D, K, K) or (K, K). Returns (T, K); row t is
    the forecast for day t+1, made at the close of day t.
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
    out = np.empty((T, K))
    for t in range(T):
        prior_t = filt if t == 0 else np.einsum("dk,dkj->dj", filt, A)
        f = prior_t * B[t]
        filt = f / f.sum(axis=1, keepdims=True)
        out[t] = np.einsum("dk,dkj->dj", filt, A).mean(axis=0)
    return out


def predictive_bull_prob(y, mu, sigma, A, bull: int = 1) -> np.ndarray:
    """P(s_{t+1} = bull | y_1..t); kept for the 2-state case (bull = calm state 1)."""
    return predictive_state_prob(y, mu, sigma, A)[:, bull]


def normalized_entropy(p) -> np.ndarray:
    """Entropy of each row of p divided by log K: 0 = certain, 1 = uniform."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1) / np.log(p.shape[1])


def exposure(p_next, rule: str) -> np.ndarray:
    """Positions from (T, K) regime forecasts; state 0 is the turbulent regime."""
    p_turb = p_next[:, 0]
    if rule == "hard":
        return (p_turb < 0.5).astype(float)
    if rule == "soft":
        return 1.0 - p_turb
    if rule == "soft_confident":
        return (1.0 - p_turb) * (1.0 - normalized_entropy(p_next))
    raise ValueError(rule)


def performance(returns, periods: int = PERIODS) -> dict:
    """Annualized Sharpe (zero risk-free rate), annualized return and volatility,
    total log return, and maximum drawdown of daily log returns."""
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    wealth = np.exp(np.cumsum(r))
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return {"sharpe": float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0,
            "ann_return": float(np.expm1(r.mean() * periods)),
            "ann_vol": float(sd * np.sqrt(periods)),
            "total_log_return": float(r.sum()),
            "max_drawdown": float((wealth / peak - 1).min())}


def strategy_returns(y, positions, cost_bps: float = 2.0) -> np.ndarray:
    """Daily log returns of holding ``positions[t]`` during day t+1, minus
    ``cost_bps`` basis points per unit of turnover. Length len(y) - 1."""
    y = np.asarray(y, dtype=float)
    held = np.asarray(positions, dtype=float)[:-1]
    turnover = np.abs(np.diff(np.concatenate([[held[0]], held])))
    return held * y[1:] - cost_bps * 1e-4 * turnover


def backtest(y, positions, cost_bps: float = 2.0) -> dict:
    """Apply positions (decided the day before) to log returns y.

    positions[t] is the exposure held during day t+1. Strategy log return is
    approximated by position * y (fine for daily returns), minus a cost of
    ``cost_bps`` basis points per unit of turnover.
    """
    held = np.asarray(positions, dtype=float)[:-1]
    res = performance(strategy_returns(y, positions, cost_bps))
    res["avg_exposure"] = float(held.mean())
    res["turnover_per_year"] = float(np.abs(np.diff(held)).sum() / len(held) * PERIODS)
    return res


def vol_target_positions(y, target_ann_vol: float, window: int = 20, cap: float = 1.0) -> np.ndarray:
    """Volatility targeting: exposure = target / trailing realized vol, capped.
    positions[t] uses returns up to and including day t. The simple benchmark
    any regime rule that de-risks in turbulence has to beat."""
    y = np.asarray(y, dtype=float)
    pos = np.full(len(y), cap)
    sq = np.concatenate([[0.0], np.cumsum(y ** 2)])
    for t in range(window - 1, len(y)):
        rv = np.sqrt((sq[t + 1] - sq[t + 1 - window]) / window * PERIODS)
        pos[t] = min(cap, target_ann_vol / rv) if rv > 0 else cap
    return pos


def block_bootstrap_sharpe_diff(a, b, n_boot: int = 2000, block: int = 20, seed: int = 0) -> dict:
    """Stationary-block bootstrap 95% interval for Sharpe(a) - Sharpe(b) of two
    aligned daily return series. Blocks keep volatility clustering, which an
    i.i.d. bootstrap would destroy (and so understate the uncertainty)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = len(a)
    rng = np.random.default_rng(seed)

    def sharpe(x):
        sd = x.std(ddof=1)
        return x.mean() / sd * np.sqrt(PERIODS) if sd > 0 else 0.0

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.empty(n, dtype=int)
        t = 0
        while t < n:
            start, length = rng.integers(n), rng.geometric(1 / block)
            take = min(length, n - t)
            idx[t:t + take] = (start + np.arange(take)) % n
            t += take
        diffs[i] = sharpe(a[idx]) - sharpe(b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(sharpe(a) - sharpe(b)), "ci95": [float(lo), float(hi)]}

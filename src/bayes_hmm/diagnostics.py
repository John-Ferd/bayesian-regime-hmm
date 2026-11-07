"""
Convergence diagnostics, from scratch.

* ``autocorrelation`` -- FFT-based sample autocorrelation of one chain.
* ``ess`` -- effective sample size across chains, n_total / (1 + 2 sum rho_k),
  truncating the sum with Geyer's initial positive sequence (pairs of lags
  are summed until a pair goes negative).
* ``split_rhat`` -- Gelman-Rubin potential scale reduction on chains split in
  half, so a chain that drifts within itself is also caught. Values near 1.00
  indicate the chains agree; > 1.01 is the modern warning threshold.
"""
from __future__ import annotations

import numpy as np


def autocorrelation(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    acov = np.fft.irfft(f * np.conj(f))[:n] / n
    return acov / acov[0] if acov[0] > 0 else np.ones(n)


def ess(chains: np.ndarray) -> float:
    """ESS of one scalar quantity; ``chains`` has shape (m, n) or (n,)."""
    chains = np.atleast_2d(np.asarray(chains, dtype=float))
    m, n = chains.shape
    rho = np.mean([autocorrelation(c) for c in chains], axis=0)
    tau = -1.0                    # 1 + 2 * sum_{k>=1} rho_k == -1 + 2 * sum_{k>=0} rho_k
    for k in range(0, n - 1, 2):
        pair = rho[k] + rho[k + 1]
        if pair < 0:
            break
        tau += 2 * pair
    return float(m * n / max(tau, 1e-12))


def split_rhat(chains: np.ndarray) -> float:
    """Split R-hat of one scalar quantity; ``chains`` has shape (m, n)."""
    chains = np.asarray(chains, dtype=float)
    half = chains.shape[1] // 2
    s = np.vstack([chains[:, :half], chains[:, half:2 * half]])
    m, n = s.shape
    means = s.mean(axis=1)
    W = s.var(axis=1, ddof=1).mean()
    B = n * means.var(ddof=1)
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def summarize(draws: np.ndarray, names, truth=None, level: float = 0.95) -> list[dict]:
    """Per-parameter summary. ``draws`` has shape (m chains, n draws, d params)."""
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    rows = []
    for j, name in enumerate(names):
        c = draws[:, :, j]
        flat = c.ravel()
        row = {
            "param": name,
            "mean": float(flat.mean()),
            "sd": float(flat.std(ddof=1)),
            "ci_lo": float(np.quantile(flat, lo)),
            "ci_hi": float(np.quantile(flat, hi)),
            "rhat": split_rhat(c),
            "ess": ess(c),
        }
        if truth is not None:
            row["truth"] = float(truth[j])
            row["covered"] = bool(row["ci_lo"] <= truth[j] <= row["ci_hi"])
        rows.append(row)
    return rows

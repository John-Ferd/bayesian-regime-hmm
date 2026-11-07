"""
Gibbs sampler for a K-state Gaussian HMM with conjugate priors.

Model
    s_1 ~ Uniform,  s_t | s_{t-1} ~ A[s_{t-1}, :],  y_t | s_t = k ~ N(mu_k, sigma2_k)

Priors (conjugate, so every Gibbs step is an exact draw)
    A[k, :]            ~ Dirichlet(alpha_k)          sticky: extra mass on A[k, k]
    sigma2_k           ~ InvGamma(a0, b0)
    mu_k | sigma2_k    ~ N(m0, sigma2_k / kappa0)    (Normal-Inverse-Gamma)

One sweep
    1. s_{1:T} | A, mu, sigma2, y   by forward-filtering backward-sampling (FFBS):
       filter P(s_t | y_1..t) forward, then sample s_T, s_{T-1}, ... backward from
       P(s_t | s_{t+1}, y_1..t)  proportional to  P(s_t | y_1..t) * A[s_t, s_{t+1}].
       This draws the whole path jointly, which mixes far better than updating
       one s_t at a time (neighbouring states are strongly correlated).
    2. A[k, :] | s       ~ Dirichlet(alpha_k + transition counts out of k)
    3. (mu_k, sigma2_k) | s, y  ~ Normal-Inverse-Gamma posterior of the y_t with s_t = k

Label switching: the likelihood is invariant to permuting state labels, so an
unconstrained sampler can swap them mid-chain and the posterior summaries
become meaningless. After every sweep the states are relabelled so that
sigma_0 > sigma_1 (state 0 = high-volatility regime). Ordering by volatility
rather than by mean is deliberate: daily means are small next to their
posterior uncertainty, so a mean ordering would itself flip, while the
volatilities differ by a wide margin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Prior:
    m0: float = 0.0
    kappa0: float = 0.01      # weak: worth 1% of one observation
    a0: float = 2.0
    b0: float = 1e-4          # InvGamma scale on the variance of daily returns (prior mode ~ (0.6%)^2)
    alpha_stay: float = 10.0  # Dirichlet mass on staying in the same regime
    alpha_move: float = 1.0


def _gauss_lik(y, mu, sigma2):
    """(T, K) Gaussian densities, rescaled per row (only ratios matter for FFBS)."""
    logp = -0.5 * np.log(sigma2)[None, :] - 0.5 * (y[:, None] - mu[None, :]) ** 2 / sigma2[None, :]
    logp -= logp.max(axis=1, keepdims=True)
    return np.exp(logp)


def ffbs(y, A, mu, sigma2, rng):
    """One joint draw of the state path s_{1:T}."""
    T, K = len(y), len(mu)
    B = _gauss_lik(y, mu, sigma2)
    filt = np.empty((T, K))
    f = np.full(K, 1.0 / K) * B[0]
    filt[0] = f / f.sum()
    for t in range(1, T):
        f = (filt[t - 1] @ A) * B[t]
        filt[t] = f / f.sum()
    s = np.empty(T, dtype=int)
    u = rng.random(T)
    s[-1] = np.searchsorted(np.cumsum(filt[-1]), u[-1] * filt[-1].sum())
    for t in range(T - 2, -1, -1):
        w = filt[t] * A[:, s[t + 1]]
        c = np.cumsum(w)
        s[t] = min(np.searchsorted(c, u[t] * c[-1]), K - 1)
    return s


def nig_posterior(yk, prior: Prior):
    """Normal-Inverse-Gamma posterior (m_n, kappa_n, a_n, b_n) given the points yk."""
    n = len(yk)
    ybar = float(np.mean(yk)) if n else 0.0
    ss = float(((yk - ybar) ** 2).sum()) if n else 0.0
    kn = prior.kappa0 + n
    mn = (prior.kappa0 * prior.m0 + n * ybar) / kn
    an = prior.a0 + n / 2
    bn = prior.b0 + 0.5 * ss + 0.5 * prior.kappa0 * n * (ybar - prior.m0) ** 2 / kn
    return mn, kn, an, bn


def _relabel(A, mu, sigma2, s):
    order = np.argsort(-sigma2)              # state 0 = highest volatility
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return A[np.ix_(order, order)], mu[order], sigma2[order], inv[s]


def gibbs_hmm(y, K: int = 2, n_iter: int = 2000, burn_in: int = 500, prior: Prior | None = None,
              seed: int = 0, init: dict | None = None):
    """Run one chain. Returns a dict of draws: mu (n, K), sigma (n, K), A (n, K, K),
    and the posterior mean of P(s_t = k | y) as ``state_prob`` (T, K)."""
    prior = prior or Prior()
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    T = len(y)
    if init is None:
        sd = y.std()
        init = {"mu": rng.normal(y.mean(), sd / 10, K),
                "sigma2": (sd * np.exp(rng.normal(0, 0.5, K))) ** 2,
                "A": np.full((K, K), 0.05 / (K - 1)) + np.eye(K) * (0.95 - 0.05 / (K - 1))}
    mu, sigma2, A = (np.asarray(init[k], dtype=float).copy() for k in ("mu", "sigma2", "A"))
    alpha = np.full((K, K), prior.alpha_move) + np.eye(K) * (prior.alpha_stay - prior.alpha_move)

    out = {"mu": np.empty((n_iter, K)), "sigma": np.empty((n_iter, K)), "A": np.empty((n_iter, K, K))}
    state_prob = np.zeros((T, K))
    for it in range(burn_in + n_iter):
        s = ffbs(y, A, mu, sigma2, rng)
        # transition matrix rows
        counts = np.zeros((K, K))
        np.add.at(counts, (s[:-1], s[1:]), 1)
        A = np.vstack([rng.dirichlet(alpha[k] + counts[k]) for k in range(K)])
        # emission parameters: Normal-Inverse-Gamma update per state
        for k in range(K):
            mn, kn, an, bn = nig_posterior(y[s == k], prior)
            sigma2[k] = bn / rng.gamma(an)                 # InvGamma(an, bn)
            mu[k] = rng.normal(mn, np.sqrt(sigma2[k] / kn))
        A, mu, sigma2, s = _relabel(A, mu, sigma2, s)
        if it >= burn_in:
            j = it - burn_in
            out["mu"][j], out["sigma"][j], out["A"][j] = mu, np.sqrt(sigma2), A
            state_prob[np.arange(T), s] += 1
    out["state_prob"] = state_prob / n_iter
    return out

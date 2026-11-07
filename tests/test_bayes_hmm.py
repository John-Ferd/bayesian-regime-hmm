import numpy as np

from bayes_hmm import Prior, backtest, gibbs_hmm, predictive_bull_prob, simulate_regimes
from bayes_hmm.backtest import performance
from bayes_hmm.diagnostics import ess, split_rhat
from bayes_hmm.gibbs import _relabel, ffbs, nig_posterior

A = np.array([[0.9, 0.1], [0.05, 0.95]])
MU = np.array([-0.01, 0.005])
S2 = np.array([0.02, 0.008]) ** 2


def _smoothed(y):
    """Exact P(s_t | y_1..T) by scaled forward-backward (uniform initial state)."""
    T = len(y)
    B = np.exp(-0.5 * np.log(S2) - 0.5 * (y[:, None] - MU) ** 2 / S2)
    a = np.empty((T, 2))
    a[0] = 0.5 * B[0]
    a[0] /= a[0].sum()
    for t in range(1, T):
        a[t] = (a[t - 1] @ A) * B[t]
        a[t] /= a[t].sum()
    b = np.ones((T, 2))
    for t in range(T - 2, -1, -1):
        b[t] = A @ (B[t + 1] * b[t + 1])
        b[t] /= b[t].sum()
    g = a * b
    return g / g.sum(axis=1, keepdims=True)


def test_ffbs_marginals_match_exact_smoothing():
    y, _ = simulate_regimes(T=60, mu=MU, sigma=np.sqrt(S2), A=A, seed=0)
    rng = np.random.default_rng(1)
    freq = np.mean([ffbs(y, A, MU, S2, rng) for _ in range(4000)], axis=0)
    np.testing.assert_allclose(freq, _smoothed(y)[:, 1], atol=0.03)


def test_nig_posterior_matches_hand_calculation():
    prior = Prior(m0=0.0, kappa0=1.0, a0=2.0, b0=1.0)
    yk = np.array([1.0, 2.0, 3.0])
    mn, kn, an, bn = nig_posterior(yk, prior)
    assert (mn, kn, an) == (1.5, 4.0, 3.5)
    assert np.isclose(bn, 1.0 + 0.5 * 2.0 + 0.5 * 1.0 * 3 * 4.0 / 4.0)


def test_relabel_orders_by_volatility_and_permutes_consistently():
    A2 = np.array([[0.9, 0.1], [0.2, 0.8]])
    A3, mu, s2, s = _relabel(A2, np.array([0.1, -0.1]), np.array([1.0, 4.0]), np.array([0, 1, 1]))
    np.testing.assert_array_equal(s2, [4.0, 1.0])
    np.testing.assert_array_equal(mu, [-0.1, 0.1])
    np.testing.assert_array_equal(A3, [[0.8, 0.2], [0.1, 0.9]])
    np.testing.assert_array_equal(s, [1, 0, 0])


def test_gibbs_recovers_volatilities_and_persistence():
    y, s = simulate_regimes(T=1500, seed=3)
    out = gibbs_hmm(y, n_iter=300, burn_in=150, seed=0)
    np.testing.assert_allclose(out["sigma"].mean(0), [0.020, 0.008], rtol=0.1)
    assert out["A"][:, 1, 1].mean() > 0.97
    assert (out["state_prob"].argmax(1) == s).mean() > 0.9


def test_predictive_probability_is_causal():
    y, _ = simulate_regimes(T=300, seed=4)
    p_short = predictive_bull_prob(y[:200], MU, np.sqrt(S2), A)
    p_long = predictive_bull_prob(y, MU, np.sqrt(S2), A)
    np.testing.assert_allclose(p_long[:200], p_short)


def test_averaging_identical_draws_equals_single_draw():
    y, _ = simulate_regimes(T=100, seed=5)
    one = predictive_bull_prob(y, MU, np.sqrt(S2), A)
    many = predictive_bull_prob(y, np.tile(MU, (5, 1)), np.tile(np.sqrt(S2), (5, 1)), np.tile(A, (5, 1, 1)))
    np.testing.assert_allclose(one, many)


def test_backtest_uses_yesterdays_position():
    y = np.array([0.0, 0.01, -0.02, 0.03])
    res = backtest(y, np.ones(4), cost_bps=0.0)
    assert np.isclose(res["total_log_return"], y[1:].sum())
    # all-in only on the day *after* a signal: position [0,1,0,0] earns y[2]
    res = backtest(y, np.array([0.0, 1.0, 0.0, 0.0]), cost_bps=0.0)
    assert np.isclose(res["total_log_return"], y[2])


def test_max_drawdown():
    r = np.log([1.1, 0.5, 1.2])          # wealth 1.1 -> 0.55 -> 0.66
    assert np.isclose(performance(r)["max_drawdown"], 0.55 / 1.1 - 1)


def test_diagnostics_on_iid_draws():
    x = np.random.default_rng(0).normal(size=(4, 2000))
    assert split_rhat(x) < 1.01
    assert 0.85 * 8000 < ess(x) < 1.15 * 8000

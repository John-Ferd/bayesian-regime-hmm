"""Bayesian 2-regime Gaussian HMM fitted by Gibbs sampling with
forward-filtering backward-sampling (FFBS)."""

from .backtest import backtest, predictive_bull_prob
from .data import load_returns, simulate_regimes
from .gibbs import Prior, gibbs_hmm

__all__ = ["Prior", "backtest", "gibbs_hmm", "load_returns", "predictive_bull_prob", "simulate_regimes"]
__version__ = "0.1.0"

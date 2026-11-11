"""Bayesian regime-switching HMM on real SPY daily returns, 2000 to October 2026.

Three periods, used strictly in order:
    train       2000-2012   the first models are fit on this
    validation  2013-2018   choose the configuration (number of regimes, point vs
                            Bayesian forecast, exposure rule)
    test        2019-2026   report once, with nothing chosen on it

Walk-forward: for every year from 2013 on, run the Gibbs sampler on all data
before that year, then trade the year using only the regime forecast made at
each previous close (forward filtering; parameters frozen for the year).
"point" plugs in the posterior-mean parameters; "bayes" averages the forecast
over 200 posterior draws. Baselines: buy-and-hold and 20-day volatility
targeting. 2 bp cost per unit of turnover.

Separately, 4 chains on the full sample give the posterior summary and the
convergence diagnostics (split R-hat, ESS). That fit is descriptive only; no
trading decision uses it.

Writes artifacts/results_SPY.json and plots.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayes_hmm import gibbs_hmm, load_returns
from bayes_hmm.backtest import (
    backtest,
    block_bootstrap_sharpe_diff,
    exposure,
    predictive_state_prob,
    strategy_returns,
    vol_target_positions,
)
from bayes_hmm.diagnostics import summarize

TRAIN_END, VAL_END = 2012, 2018
RULES = ("hard", "soft", "soft_confident")


def _walk_forward_year(task):
    """Fit on y_fit, then forecast regimes for days lo..hi of y_all (both point and Bayesian)."""
    y_all, n_fit, lo, hi, K, n_iter, burn_in, seed = task
    out = gibbs_hmm(y_all[:n_fit], K=K, n_iter=n_iter, burn_in=burn_in, seed=seed)
    idx = np.random.default_rng(seed).choice(n_iter, size=min(200, n_iter), replace=False)
    mu_pt, sg_pt, A_pt = out["mu"].mean(0), out["sigma"].mean(0), out["A"].mean(0)
    p_point = predictive_state_prob(y_all[:hi + 1], mu_pt, sg_pt, A_pt)[lo:hi + 1]
    p_bayes = predictive_state_prob(y_all[:hi + 1], out["mu"][idx], out["sigma"][idx], out["A"][idx])[lo:hi + 1]
    return p_point, p_bayes, {"mu": mu_pt.tolist(), "sigma": sg_pt.tolist(), "p_stay": np.diag(A_pt).tolist()}


def _chain(task):
    y, K, n_iter, burn_in, seed = task
    return gibbs_hmm(y, K=K, n_iter=n_iter, burn_in=burn_in, seed=seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--csv", default=None, help="daily price CSV (Date + Adj Close/Close); default: data/<ticker>.csv")
    ap.add_argument("--n-iter", type=int, default=1000, help="kept draws per walk-forward fit")
    ap.add_argument("--burn-in", type=int, default=500)
    ap.add_argument("--chains", type=int, default=4, help="chains for the full-sample diagnostic fit")
    ap.add_argument("--diag-iter", type=int, default=2000)
    ap.add_argument("--cost-bps", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "artifacts"
    out_dir.mkdir(exist_ok=True)

    y, dates = load_returns(args.ticker, csv=args.csv, cache_dir=root / "data")
    year = dates.year.to_numpy()
    va = (year > TRAIN_END) & (year <= VAL_END)
    te = year > VAL_END
    years = list(range(TRAIN_END + 1, int(year.max()) + 1))

    # ------------------------------------------------------------ walk-forward fits
    tasks, keys = [], []
    for K in (2, 3):
        for Y in years:
            days = np.flatnonzero(year == Y)
            lo, hi = days[0] - 1, days[-1]          # forecasts from the last close before Y to the end of Y
            tasks.append((y, days[0], lo, hi, K, args.n_iter, args.burn_in, args.seed + Y))
            keys.append((K, Y, lo, hi))
    with ProcessPoolExecutor(args.workers) as ex:
        fitted = list(ex.map(_walk_forward_year, tasks))
        diag = list(ex.map(_chain, [(y, 2, args.diag_iter, args.burn_in, args.seed + 100 + c)
                                    for c in range(args.chains)]
                           + [(y, 3, args.diag_iter, args.burn_in, args.seed + 200 + c) for c in range(args.chains)]))

    p_next = {(K, est): np.full((len(y), K), np.nan) for K in (2, 3) for est in ("point", "bayes")}
    yearly_params = {2: {}, 3: {}}
    for (K, Y, lo, hi), (pp, pb, params) in zip(keys, fitted):
        p_next[(K, "point")][lo:hi + 1] = pp
        p_next[(K, "bayes")][lo:hi + 1] = pb
        yearly_params[K][Y] = {k: [round(v, 5) for v in vals] for k, vals in params.items()}

    target = float(y[year <= TRAIN_END].std() * np.sqrt(252))
    pos = {"buy_and_hold": np.ones(len(y)), "vol_target_20d": vol_target_positions(y, target)}
    configs = []
    for (K, est), p in p_next.items():
        for rule in RULES:
            name = f"K{K}_{est}_{rule}"
            filled = np.where(np.isnan(p).any(axis=1, keepdims=True), np.full_like(p, 1.0 / K), p)
            pos[name] = exposure(filled, rule)
            configs.append(name)

    def period_slice(mask):
        idx = np.flatnonzero(mask)
        return slice(idx[0] - 1, idx[-1] + 1)

    def evaluate(mask):
        s = period_slice(mask)
        return {k: {kk: round(v, 4) for kk, v in backtest(y[s], p[s], args.cost_bps).items()} for k, p in pos.items()}

    bt_val, bt_test = evaluate(va), evaluate(te)
    chosen = max(configs, key=lambda k: bt_val[k]["sharpe"])
    s = period_slice(te)
    r = {k: strategy_returns(y[s], pos[k][s], args.cost_bps) for k in pos}
    boot = {base: block_bootstrap_sharpe_diff(r[chosen], r[base], seed=args.seed)
            for base in ("buy_and_hold", "vol_target_20d")}
    bayes_v, point_v = chosen.replace("_point_", "_bayes_"), chosen.replace("_bayes_", "_point_")
    boot["bayes_minus_point_same_K_and_rule"] = block_bootstrap_sharpe_diff(r[bayes_v], r[point_v], seed=args.seed)

    # ------------------------------------------------------------ full-sample posterior + diagnostics
    posterior = {}
    for K, chains in ((2, diag[:args.chains]), (3, diag[args.chains:])):
        names = [f"mu_{k}" for k in range(K)] + [f"ann_vol_{k}" for k in range(K)] + [f"p_stay_{k}" for k in range(K)]
        draws = np.stack([np.column_stack([ch["mu"], ch["sigma"] * np.sqrt(252),
                                           np.stack([ch["A"][:, k, k] for k in range(K)], axis=1)])
                          for ch in chains])
        posterior[K] = [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}
                        for row in summarize(draws, names)]

    def rnd(d):
        return {k: (round(v, 3) if isinstance(v, float) else [round(c, 3) for c in v]) for k, v in d.items()}

    res = {
        "data": f"{args.ticker} daily log returns of adjusted closes, {dates[0].date()} to {dates[-1].date()}",
        "periods": {"train": f"2000-{TRAIN_END}", "validation": f"{TRAIN_END + 1}-{VAL_END}",
                    "test": f"{VAL_END + 1}-{dates[-1].year}"},
        "n_days": {"validation": int(va.sum()), "test": int(te.sum())},
        "walk_forward": {"refit": "yearly, expanding window from 2000", "draws_per_fit": args.n_iter,
                         "burn_in": args.burn_in, "posterior_draws_averaged": 200},
        "cost_bps_per_unit_turnover": args.cost_bps,
        "vol_target_ann": round(target, 4),
        "states": "ordered by volatility: state 0 = most turbulent",
        "config_chosen_on_validation": chosen,
        "validation": bt_val,
        "test": bt_test,
        "test_sharpe_diff_block_bootstrap": {k: rnd(v) for k, v in boot.items()},
        "posterior_full_sample_4_chains": {f"K{K}": rows for K, rows in posterior.items()},
        "yearly_posterior_means": {f"K{K}": v for K, v in yearly_params.items()},
    }
    (out_dir / f"results_{args.ticker}.json").write_text(json.dumps(res, indent=2))

    print(f"chosen on validation: {chosen}")
    print(f"{'strategy':26s} {'val Sharpe':>10s} {'test Sharpe':>11s} {'test ann ret':>12s} {'test maxDD':>10s} "
          f"{'exposure':>8s}")
    for k in ["buy_and_hold", "vol_target_20d"] + configs:
        v, t = bt_val[k], bt_test[k]
        print(f"{k:26s} {v['sharpe']:10.2f} {t['sharpe']:11.2f} {t['ann_return']:12.1%} {t['max_drawdown']:10.1%} "
              f"{t['avg_exposure']:8.2f}")
    print(json.dumps(res["test_sharpe_diff_block_bootstrap"], indent=1))
    for K, rows in posterior.items():
        print(f"K={K}: max R-hat {max(r_['rhat'] for r_ in rows):.3f}, min ESS {min(r_['ess'] for r_ in rows):.0f}")

    # ------------------------------------------------------------ plots
    K = int(chosen[1])
    chains = diag[:args.chains] if K == 2 else diag[args.chains:]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ch in chains:
        axes[0].plot(ch["sigma"][:, 0] * np.sqrt(252), lw=0.5, alpha=0.8)
        axes[1].plot(ch["A"][:, 0, 0], lw=0.5, alpha=0.8)
    axes[0].set_title(f"K={K}: turbulent-regime annual vol, {len(chains)} chains")
    axes[1].set_title(f"K={K}: P(stay turbulent), {len(chains)} chains")
    for ax in axes:
        ax.set_xlabel("draw")
    fig.tight_layout()
    fig.savefig(out_dir / f"traces_{args.ticker}.png", dpi=130)
    plt.close(fig)

    est = "bayes" if "_bayes_" in chosen else "point"
    p_turb = p_next[(K, est)][:, 0]
    px = np.exp(np.cumsum(y))
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=False, gridspec_kw={"height_ratios": [2, 1, 2]})
    wf = ~np.isnan(p_turb)
    axes[0].semilogy(dates[wf], px[wf], color="k", lw=0.8)
    axes[0].fill_between(dates[wf], px[wf].min(), px[wf].max(), where=p_turb[wf] > 0.5, color="tab:red",
                         alpha=0.18, lw=0, label="P(turbulent tomorrow) > 0.5, from data up to today")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel(f"{args.ticker} (log scale)")
    axes[0].set_title(f"Walk-forward regime forecast, {chosen}")
    axes[1].plot(dates[wf], pos[chosen][wf], lw=0.6)
    axes[1].set_ylabel("position")
    axes[1].axvline(pd.Timestamp(f"{VAL_END + 1}-01-01"), color="gray", ls="--", lw=0.8)
    for k in ("buy_and_hold", "vol_target_20d", chosen):
        axes[2].plot(dates[s][1:], np.exp(np.cumsum(r[k])), lw=1, label=f"{k} (Sharpe {bt_test[k]['sharpe']:.2f})")
    axes[2].set_title(f"Test period {VAL_END + 1}-{dates[-1].year}: growth of $1")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"backtest_{args.ticker}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()

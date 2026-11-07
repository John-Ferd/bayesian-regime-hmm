"""Fit the Bayesian regime HMM and backtest regime-filtered positions.

1. Simulate a training series with known regimes (2,000 days by default;
   --train-days 250 shows what changes when data is scarce).
2. Run 4 Gibbs chains (FFBS) from dispersed starts; report posterior summaries,
   95% credible intervals vs the truth, split R-hat and ESS.
3. Backtest on 20 *new* 1,000-day paths from the same process, so the result
   is a distribution over paths rather than one lucky path. Positions are
   decided causally from data up to the previous close.

With --ticker the model is fit to real returns instead (no truth, no backtest
paths; the last 30% of the series is the out-of-sample backtest).
Writes artifacts/results.json and plots.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bayes_hmm import backtest, gibbs_hmm, load_returns, predictive_bull_prob, simulate_regimes
from bayes_hmm.diagnostics import summarize

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--ticker", default=None)
ap.add_argument("--csv", default=None, help="daily price CSV (Date + Adj Close/Close); use with --ticker for the name")
ap.add_argument("--chains", type=int, default=4)
ap.add_argument("--n-iter", type=int, default=2000)
ap.add_argument("--burn-in", type=int, default=500)
ap.add_argument("--paths", type=int, default=20)
ap.add_argument("--train-days", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
out_dir = Path(__file__).resolve().parents[1] / "artifacts"
out_dir.mkdir(exist_ok=True)
tag = f"_{args.ticker}" if args.ticker else ("" if args.train_days == 2000 else f"_train{args.train_days}")

TRUE = {"mu": (-0.0015, 0.0008), "sigma": (0.020, 0.008), "A": ((0.97, 0.03), (0.01, 0.99))}
if args.ticker:
    y_all, _ = load_returns(args.ticker, csv=args.csv)
    split = int(0.7 * len(y_all))
    y_train, test_paths, truth = y_all[:split], [y_all[split:]], None
else:
    y_train, _ = simulate_regimes(T=args.train_days, seed=args.seed, **TRUE)
    test_paths = [simulate_regimes(T=1000, seed=100 + i, **TRUE)[0] for i in range(args.paths)]
    truth = np.array([*TRUE["mu"], *TRUE["sigma"], TRUE["A"][0][0], TRUE["A"][1][1]])

# ---- Gibbs, several chains --------------------------------------------------
chains = [gibbs_hmm(y_train, n_iter=args.n_iter, burn_in=args.burn_in, seed=args.seed + c)
          for c in range(args.chains)]
names = ["mu_bear", "mu_bull", "sigma_bear", "sigma_bull", "p_stay_bear", "p_stay_bull"]
draws = np.stack([np.column_stack([ch["mu"], ch["sigma"], ch["A"][:, 0, 0], ch["A"][:, 1, 1]])
                  for ch in chains])                       # (chains, n_iter, 6)
summary = summarize(draws, names, truth)

# ---- causal backtest ----------------------------------------------------------
mu_d = np.concatenate([ch["mu"] for ch in chains])
sg_d = np.concatenate([ch["sigma"] for ch in chains])
A_d = np.concatenate([ch["A"] for ch in chains])
idx = np.random.default_rng(args.seed).choice(len(mu_d), size=200, replace=False)
mu_pt, sg_pt, A_pt = mu_d.mean(0), sg_d.mean(0), A_d.mean(0)   # plug-in point estimate

strategies = ["buy_and_hold", "point_hard", "point_soft", "bayes_soft"]
per_path = {k: [] for k in strategies}
for y in test_paths:
    p_pt = predictive_bull_prob(y, mu_pt, sg_pt, A_pt)
    p_bayes = predictive_bull_prob(y, mu_d[idx], sg_d[idx], A_d[idx])
    pos = {"buy_and_hold": np.ones(len(y)), "point_hard": (p_pt > 0.5).astype(float),
           "point_soft": p_pt, "bayes_soft": p_bayes}
    for k in strategies:
        per_path[k].append(backtest(y, pos[k]))


def agg(rows, key):
    v = np.array([r[key] for r in rows])
    return {"mean": round(float(v.mean()), 3), "median": round(float(np.median(v)), 3),
            "min": round(float(v.min()), 3), "max": round(float(v.max()), 3)}


bt = {k: {m: agg(per_path[k], m) for m in ("sharpe", "max_drawdown", "total_log_return",
                                           "avg_exposure", "turnover_per_year")} for k in strategies}
bayes_beats = {k: int(sum(b["sharpe"] > o["sharpe"] for b, o in zip(per_path["bayes_soft"], per_path[k])))
               for k in strategies if k != "bayes_soft"}

res = {
    "data": args.ticker or "synthetic: %d training days, %d test paths of 1,000 days" % (args.train_days, args.paths),
    "gibbs": {"chains": args.chains, "draws_per_chain": args.n_iter, "burn_in": args.burn_in},
    "posterior": [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in summary],
    "backtest": bt,
    "bayes_soft_higher_sharpe_than": {k: f"{v}/{len(test_paths)} paths" for k, v in bayes_beats.items()},
    "cost_bps_per_unit_turnover": 1.0,
}
(out_dir / f"results{tag}.json").write_text(json.dumps(res, indent=2))

print(f"{'param':12s} {'truth':>8s} {'mean':>9s} {'95% CI':>22s} {'R-hat':>6s} {'ESS':>6s} cover")
for r in summary:
    t = r.get("truth", float("nan"))
    print(f"{r['param']:12s} {t:8.4f} {r['mean']:9.5f} [{r['ci_lo']:9.5f},{r['ci_hi']:9.5f}] "
          f"{r['rhat']:6.3f} {r['ess']:6.0f} {r.get('covered', '')}")
print()
for k in strategies:
    b = bt[k]
    print(f"{k:13s} Sharpe mean {b['sharpe']['mean']:6.2f} (median {b['sharpe']['median']:5.2f})  "
          f"maxDD mean {b['max_drawdown']['mean']:6.1%}  worst {b['max_drawdown']['min']:6.1%}  "
          f"exposure {b['avg_exposure']['mean']:.2f}  turnover/yr {b['turnover_per_year']['mean']:.1f}")
print("bayes_soft higher Sharpe than:", res["bayes_soft_higher_sharpe_than"])

# ---- plots --------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
for c in range(args.chains):
    axes[0].plot(draws[c, :, 2], lw=0.5, alpha=0.8)
    axes[1].plot(draws[c, :, 5], lw=0.5, alpha=0.8)
for ax, j in ((axes[0], 2), (axes[1], 5)):
    if truth is not None:
        ax.axhline(truth[j], color="k", ls="--", lw=1)
    ax.set_title(f"{names[j]}: {args.chains} chains")
    ax.set_xlabel("draw")
fig.tight_layout()
fig.savefig(out_dir / f"traces{tag}.png", dpi=130)
plt.close(fig)

y = test_paths[0]
p_pt = predictive_bull_prob(y, mu_pt, sg_pt, A_pt)
p_bayes = predictive_bull_prob(y, mu_d[idx], sg_d[idx], A_d[idx])
fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
for k, w in [("buy and hold", np.ones(len(y))), ("point HMM, hard switch", (p_pt > 0.5).astype(float)),
             ("Bayesian HMM, soft", p_bayes)]:
    axes[0].plot(np.exp(np.cumsum(w[:-1] * y[1:])), label=k, lw=1)
axes[0].set_ylabel("wealth")
axes[0].legend(fontsize=8)
axes[0].set_title("Test path 1: growth of $1")
axes[1].plot(p_bayes, lw=0.8, label="Bayesian P(bull tomorrow | data to today)")
axes[1].set_ylabel("position")
axes[1].set_xlabel("day")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(out_dir / f"backtest{tag}.png", dpi=130)
plt.close(fig)

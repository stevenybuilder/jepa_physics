"""Audit robustness checks (paper-vs-code procedural deviations, Part 1): helpers for scripts/run_audit_robustness.py.

Grouped CV folds (App. B "5-fold grouped cross-validation", key unstated), a binned-accuracy score beside R² for the
sawtooth test (Fig. 4c "probe accuracy"), the per-round decay statistics of wm.inlp.sawtooth for any metric, and a
probe-sequence runner that records every key of the score dict. Everything else is reused from wm.probes / wm.inlp /
wm.steer / wm.paperscale.
"""
import numpy as np

from wm.inlp import isolated_dips, probe_sequence
from wm.paperscale import random_folds
from wm.probes import predict, score

N_BINS = 8


def start_cells(x, y, cell=1.0, lo=-2.0):
    """Integer cell id of each start position on a square grid of side `cell` metres anchored at `lo`."""
    ix = np.floor((np.asarray(x, float) - lo) / cell).astype(int)
    iy = np.floor((np.asarray(y, float) - lo) / cell).astype(int)
    return ix * 1000 + iy


def value_grouped_folds(values, n_folds=5, seed=0):
    """Fold per row with every distinct value (a label value or a start cell) held out whole: distinct values are
    shuffled and dealt round-robin to the folds (wm.paperscale.random_folds with groups = values)."""
    values = np.round(np.asarray(values, float), 6)
    assert len(np.unique(values)) >= n_folds, "fewer groups than folds"
    return random_folds(len(values), n_folds, seed, groups=values)


def direction_bin(theta_deg):
    """8-way 45° bin, edges at 0, 45, ..., 315 (the 64 directions fall 8 per bin)."""
    return (np.floor(np.mod(np.asarray(theta_deg, float), 360.0) / 45.0).astype(int)) % N_BINS


def equal_count_edges(values, n_bins=N_BINS):
    """Inner edges (n_bins − 1) of equal-count bins of `values` (label quantiles; label-only, no features)."""
    return np.quantile(np.asarray(values, float), np.linspace(0, 1, n_bins + 1)[1:-1])


def binned_score(kind, edges=None):
    """score_fn: wm.probes.score plus 'bacc8', the fraction of clips whose prediction lands in the true value's bin.
    circular: 45° bins of atan2(sin, cos); scalar: bins cut at `edges` (equal_count_edges of the train labels)."""
    def fn(Y, P):
        Y, P = np.asarray(Y, float).reshape(len(Y), -1), np.asarray(P, float).reshape(len(P), -1)
        out = score(Y, P, kind)
        if kind == "circular":
            bt = direction_bin(np.degrees(np.arctan2(Y[:, 0], Y[:, 1])))
            bp = direction_bin(np.degrees(np.arctan2(P[:, 0], P[:, 1])))
        else:
            bt, bp = np.digitize(Y[:, 0], edges), np.digitize(P[:, 0], edges)
        out["bacc8"] = float(np.mean(bt == bp))
        return out
    return fn


def decay_stats(values):
    """wm.inlp.sawtooth's per-metric statistics for a higher-is-better curve: per-round drops, fraction of rises,
    lag-1 autocorrelation of the drops (negative = alternating big/small = sawtooth) and isolated dips."""
    drops = -np.diff(np.asarray(values, float))
    return {"drop_lag1_autocorr": float(np.corrcoef(drops[:-1], drops[1:])[0, 1]) if len(drops) > 2 else None,
            "frac_rises": float(np.mean(drops < 0)) if len(drops) else None,
            "isolated_dips": isolated_dips(values), "n_isolated_dips": len(isolated_dips(values)),
            "n_rounds": len(values)}


def run_rounds(Xfit, Yfit, fit, score_fn, Xeval, Yeval, n_rounds):
    """n_rounds of wm.inlp.probe_sequence; every key of score_fn's dict per round (fit on Xfit, scored on Xeval)."""
    seq = probe_sequence(Xfit, Yfit, fit, score_fn, Xeval, Yeval)
    return [{"round": k + 1, **next(seq)[0]} for k in range(n_rounds)]


def in_sample_and_oof(fit, X, Y, folds, kind):
    """In-sample score of fit(X, Y) and fold-mean out-of-fold R² / MAE under `folds` (for an evaluation probe)."""
    W, b = fit(X, Y)
    ins = score(Y, predict(X, W, b), kind)
    fs = []
    for k in np.unique(folds):
        Wk, bk = fit(X[folds != k], Y[folds != k])
        fs.append(score(Y[folds == k], predict(X[folds == k], Wk, bk), kind))
    return W, b, {"in_sample": {"r2": ins["r2"], "mae": ins["mae"]},
                  "out_of_fold": {"r2_fold_mean": float(np.mean([s["r2"] for s in fs])),
                                  "r2_fold_sd": float(np.std([s["r2"] for s in fs], ddof=1)),
                                  "mae_fold_mean": float(np.mean([s["mae"] for s in fs]))}}


def k_below(values, thr):
    """Probes kept before the first round whose R² falls below thr (first such round − 1, rounds 1-based); None if
    the curve never falls below thr (censored)."""
    return next((k for k, v in enumerate(values) if v < thr), None)


def lockstep_until(seqs, thr, max_rounds):
    """Advance wm.inlp.probe_sequence generators (one per fold, or one for the paper protocol) in lockstep until the
    mean R² over them falls below thr (that round included) or max_rounds. Per round: mean r2, mae, base_mae (and
    acc15 when scored) over the generators, plus each generator's r2."""
    rows = []
    for k in range(1, max_rounds + 1):
        sc = [next(g)[0] for g in seqs]
        row = {"round": k, **{key: float(np.mean([s[key] for s in sc])) for key in ("r2", "mae", "base_mae", "acc15")
                              if key in sc[0]}, "each_r2": [float(s["r2"]) for s in sc]}
        rows.append(row)
        if row["r2"] < thr:
            break
    return rows

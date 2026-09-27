"""Object permanence: is the clip's direction (or speed) still linearly decodable from time-step tokens after the
disk has left the frame?

Inputs are the stored V-JEPA time pool (f16 [N, 26, 8, 1024]: mean over the 256 spatial tokens of each of the 8
tubelet time steps) and the token-grid disk mask ([N, 8, 16, 16]; a step is "absent" when no 16x16 patch holds a
disk pixel in either of its two frames).

Per layer point: a ridge probe (wm.probes helpers) is fit on the VISIBLE time-step tokens of train clips only, with
features standardised per time step on train visible tokens; alpha is picked by 5-fold CV over those tokens with the
split's clip-level folds. It is then read out on
  (i)   visible steps of test clips (and two matched subsets: visible steps at the time steps where absences occur,
        and the visible steps of the clips that later lose the disk),
  (ii)  absent steps of test clips,
  (iii) absent steps of test clips scored against clip-shuffled labels (permutation null).
"pooled" repeats every read on test + train clips, the train clips read out by 5-fold cross-fitted probes (a clip's
tokens are never in the fit that scores it; test labels never enter a fit). The persistence curve scores absent
steps by lag after the last visible step (lag 0 = that last visible step).

Caveat carried into the results: the V-JEPA encoder attends over all 8 time steps at once (no causal mask), so an
absent-step token can read the visible steps of the same clip directly; decodability there shows the direction is
carried into the absent-step tokens, not that it is held by a memory mechanism.
"""
import numpy as np

from wm.probes import ALPHAS, cv_select_alpha, fit_ridge, predict, score

LAYERS = (1, 4, 8, 12, 16, 22, 25)


def absence(diskmask):
    """visible [N, 8] bool, trailing-absent step count [N], last visible step [N] (-1 if never visible)."""
    vis = np.asarray(diskmask).reshape(len(diskmask), diskmask.shape[1], -1).any(axis=2)
    T = vis.shape[1]
    trailing = np.array([next((k for k in range(T) if row[T - 1 - k]), T) for row in vis])
    last_vis = np.array([np.flatnonzero(row).max() if row.any() else -1 for row in vis])
    return vis, trailing, last_vis


def absence_counts(vis, trailing, tr, te):
    """Clips with the disk absent in (at least / exactly) the last k steps, overall and per split."""
    T = vis.shape[1]
    out = {"any_absent_step": int((~vis).any(axis=1).sum()),
           "non_trailing_absent_steps": int(((~vis).sum(axis=1) != trailing).sum())}
    for name, rows in (("all", np.arange(len(vis))), ("train", tr), ("test", te)):
        out[name] = {"at_least_k": {k: int((trailing[rows] >= k).sum()) for k in range(1, T + 1)
                                    if (trailing[rows] >= k).any()},
                     "exactly_k": {k: int((trailing[rows] == k).sum()) for k in range(1, T + 1)
                                   if (trailing[rows] == k).any()}}
    return out


class StepStandardizer:
    """Per time step z-score, fit on the given (clip, step) tokens only."""

    def fit(self, X, mask):
        """X [n, T, d]; mask [n, T] bool (tokens used for the statistics)."""
        T = X.shape[1]
        self.mean = np.zeros((T, X.shape[2]))
        self.std = np.ones((T, X.shape[2]))
        for t in range(T):
            rows = mask[:, t] if mask[:, t].any() else np.ones(len(X), bool)
            self.mean[t] = X[rows, t].mean(axis=0)
            self.std[t] = np.maximum(X[rows, t].std(axis=0), 1e-6)
        return self

    def transform(self, X):
        return (X - self.mean) / self.std


def _tokens(mask):
    """(clip index, step) pairs of a [N, T] bool mask."""
    return np.nonzero(mask)


def _score_nor2(Y, P, kind):
    """score() without R2 (for a set with a single label value)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {k: v for k, v in score(Y, P, kind).items() if k != "r2"}


def _score_tokens(Y, P, clips, kind, n_boot, rng):
    """Score token predictions; 95% CI by resampling clips (all of a clip's tokens move together)."""
    if len(clips) == 0:
        return {"n_clips": 0, "n_tokens": 0}
    uniq, inv = np.unique(clips, return_inverse=True)
    if len(np.unique(Y, axis=0)) < 2:              # one label value: R2 undefined; MAE only
        s = {**_score_nor2(Y, P, kind), "r2": None}
        return {"n_clips": int(len(uniq)), "n_tokens": int(len(clips)), **s,
                "note": "all tokens share one label value; R2 undefined, no CI"}
    s = score(Y, P, kind)
    groups = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    out = {"n_clips": int(len(uniq)), "n_tokens": int(len(clips)), **s}
    if len(uniq) >= 3 and n_boot:
        draws = {k: [] for k in s}
        for _ in range(n_boot):
            idx = np.concatenate([groups[g] for g in rng.integers(0, len(uniq), len(uniq))])
            if len(np.unique(Y[idx], axis=0)) < 2:
                continue
            for k, v in score(Y[idx], P[idx], kind).items():
                draws[k].append(v)
        for k, v in draws.items():
            out[f"{k}_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return out


def _shuffle_null(Y, P, clips, kind, n_perm, rng, observed):
    """Permutation null: each clip's tokens get another clip's label (clip-level shuffle within the set)."""
    if len(np.unique(clips)) < 3:
        return None
    uniq, inv = np.unique(clips, return_inverse=True)
    clip_label = np.stack([Y[np.flatnonzero(inv == g)[0]] for g in range(len(uniq))])
    maes, r2s = [], []
    for _ in range(n_perm):
        Ys = clip_label[rng.permutation(len(uniq))][inv]
        s = score(Ys, P, kind)
        maes.append(s["mae"])
        r2s.append(s["r2"])
    maes, r2s = np.array(maes), np.array(r2s)
    return {"n_perm": n_perm, "mae_mean": float(maes.mean()),
            "mae_95": [float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))],
            "r2_mean": float(r2s.mean()), "r2_95": [float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5))],
            "p_mae": float((1 + np.sum(maes <= observed["mae"])) / (n_perm + 1)),
            "p_r2": float((1 + np.sum(r2s >= observed["r2"])) / (n_perm + 1))}


def evaluate_point(X, vis, last_vis, Y, kind, tr, te, folds, n_boot=1000, n_perm=1000, seed=0):
    """One layer point. X [N, T, d] float (time-step tokens), vis [N, T], Y [N, m] clip labels, tr/te row indices,
    folds [len(tr)] clip-level folds. Returns the readouts described in the module docstring."""
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    N, T = vis.shape
    st = StepStandardizer().fit(X[tr], vis[tr])
    Z = st.transform(X)

    ci, ti = _tokens(vis[tr])
    Xfit, Yfit, ffit = Z[tr][ci, ti], Y[tr][ci], folds[ci]
    cv = cv_select_alpha(Xfit, Yfit, ffit, ALPHAS, lambda a, b: score(a, b, kind))
    W, b = fit_ridge(Xfit, Yfit, cv["alpha"])
    P = predict(Z.reshape(N * T, -1), W, b).reshape(N, T, -1)        # all-train probe, read on every token

    P_oos = P.copy()                                                   # train rows: cross-fitted
    for k in np.unique(folds):
        fit_rows = folds[ci] != k
        Wk, bk = fit_ridge(Xfit[fit_rows], Yfit[fit_rows], cv["alpha"])
        rows = tr[folds == k]
        P_oos[rows] = predict(Z[rows].reshape(len(rows) * T, -1), Wk, bk).reshape(len(rows), T, -1)

    absent = ~vis
    ever_absent = absent.any(axis=1)
    absent_steps = np.unique(np.nonzero(absent)[1])
    step_is_absent_time = np.isin(np.arange(T), absent_steps)[None, :]
    seen = last_vis >= 0                                                             # never-visible clips have no lag
    lag = np.where(absent & seen[:, None], np.arange(T)[None, :] - last_vis[:, None], -1)   # >0 on trailing absences

    def read(rows, Pm):
        in_rows = np.zeros(N, bool)
        in_rows[rows] = True
        sets = {"visible": vis & in_rows[:, None],
                "visible_time_matched": vis & step_is_absent_time & in_rows[:, None],
                "visible_same_clips": vis & (ever_absent & in_rows)[:, None],
                "absent": absent & in_rows[:, None]}
        out = {}
        for name, m in sets.items():
            c, t = _tokens(m)
            out[name] = _score_tokens(Y[c], Pm[c, t], c, kind, n_boot, rng)
        c, t = _tokens(sets["absent"])
        if out["absent"].get("n_clips", 0) >= 3 and out["absent"].get("r2") is not None:
            out["absent_shuffled"] = _shuffle_null(Y[c], Pm[c, t], c, kind, n_perm, rng, out["absent"])
        persistence = {}
        last_mask = np.zeros_like(vis)
        last_mask[np.arange(N), np.clip(last_vis, 0, T - 1)] = True
        c, t = _tokens(last_mask & (ever_absent & seen & in_rows)[:, None])
        persistence["0"] = _score_tokens(Y[c], Pm[c, t], c, kind, n_boot, rng)
        for L in range(1, T):
            c, t = _tokens((lag == L) & in_rows[:, None])
            if len(c):
                persistence[str(L)] = _score_tokens(Y[c], Pm[c, t], c, kind, n_boot, rng)
        out["persistence"] = persistence
        return out

    return {"alpha": cv["alpha"], "cv_r2_visible_train": cv["cv_mean"], "cv_mae_visible_train": cv["cv_mae_mean"],
            "n_fit_tokens": int(len(Yfit)),
            "test": read(te, P),
            "pooled": read(np.concatenate([tr, te]), P_oos)}


def run(timepool, diskmask, Y, kind, tr, te, folds, points=LAYERS, n_boot=1000, n_perm=1000, seed=0, log=print):
    """Evaluate every point; timepool [N, P, T, d] (memmap fine), diskmask [N, T, g, g]."""
    vis, trailing, last_vis = absence(diskmask)
    out = {"kind": kind, "counts": absence_counts(vis, trailing, tr, te), "points": []}
    for p in points:
        X = np.asarray(timepool[:, p], dtype=np.float32).astype(np.float64)
        r = evaluate_point(X, vis, last_vis, Y, kind, tr, te, folds, n_boot, n_perm, seed)
        r["point"] = int(p)
        out["points"].append(r)
        te_, po = r["test"], r["pooled"]
        log(f"point {p:2d}: alpha={r['alpha']:.3g}  test vis MAE={te_['visible']['mae']:.3g} "
            f"absent MAE={te_['absent'].get('mae', float('nan')):.3g} (n={te_['absent']['n_clips']})  "
            f"pooled absent MAE={po['absent'].get('mae', float('nan')):.3g} (n={po['absent']['n_clips']}) "
            f"shuffled={(po.get('absent_shuffled') or {}).get('mae_mean', float('nan')):.3g}")
    return out

"""Rotating-code detector on the speed set (the steering-lessons note, section 6 P1): is the speed axis the same at
every direction (a cylinder: speed x ring) or does it rotate with direction (a cone, or anything else where the
speed read-out direction depends on theta)?

Speed set: 64 speeds x 64 directions, meanpool at points 12 and 22. Train folds 0-4 only (the test clips are not
read); 5-fold CV over the train folds. In each CV split: PCA-64 fit on the training rows, standardised; then
  global axis  ridge (RidgeCV) of speed on all training rows
  local axes   one ridge per 22.5-degree direction bin (16 bins), on that bin's training rows
  random-bin   the same local fit on 16 random bins of the same sizes (ignoring direction): the noise null for a
               local axis fit on ~1/16 of the rows
Scores per split: cosine between each local axis and the global axis (and the random-bin null), and held-out R^2 of
speed from a 1-D projection onto (a) the global axis and (b) the clip's own bin's local axis, each followed by a
per-bin 1-D linear calibration fit on training rows, so the only difference is the axis direction. A rotating code
shows local >> global and local >> random-bin, with local-global cosine below the random-bin null.
Planted controls (on the real activations, label = the speeds permuted across clips so the real speed code carries
no information about it): rotating  a * z * (cos theta u + sin theta v), constant  a * z * u, with u, v random unit
directions and a = --plant-scale x the activations' SD along the real (ridge) speed direction, i.e. a code as
strong as the real one. The detector must flag the first
and not the second.

  python scripts/run_rotating_speed_axis.py --layers 12 22
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))  # run without installing wm
from wm.data import PROJECT_ROOT
from wm.p2_data import load_inputs
from wm.provenance import layer_role, provenance

ALPHAS = np.logspace(-1, 5, 13)


def _bins(theta, n_bins):
    return (np.floor(((np.asarray(theta) % 360.0) + 180.0 / n_bins) / (360.0 / n_bins)).astype(int)) % n_bins


def _unit(w):
    return w / (np.linalg.norm(w) + 1e-12)


def _axis(Z, y):
    return _unit(RidgeCV(alphas=ALPHAS).fit(Z, y).coef_)


def _proj_r2(Z_tr, y_tr, g_tr, Z_te, y_te, g_te, axis_of_group):
    """Held-out R^2 of y from projection onto the group's axis + per-group 1-D linear calibration (train rows)."""
    pred = np.empty(len(y_te))
    for g in np.unique(g_te):
        a = axis_of_group(g)
        ptr, pte = Z_tr[g_tr == g] @ a, Z_te[g_te == g] @ a
        A = np.stack([ptr, np.ones_like(ptr)], 1)
        coef = np.linalg.lstsq(A, y_tr[g_tr == g], rcond=None)[0]
        pred[g_te == g] = pte * coef[0] + coef[1]
    return 1.0 - ((y_te - pred) ** 2).sum() / ((y_te - y_te.mean()) ** 2).sum()


def detect(X, y, theta, fold, n_bins=16, k=64, seed=0):
    """Rotating-code detector. X [N, D] train rows, y [N] speed, theta [N] degrees, fold [N] CV folds.
    Returns per-split lists and their means: cos_local_global, cos_random_global, r2_global, r2_local, r2_random."""
    rng = np.random.default_rng(seed)
    bins = _bins(theta, n_bins)
    res = {q: [] for q in ("cos_local_global", "cos_random_global", "r2_global", "r2_local", "r2_random",
                           "cos_local_global_per_bin")}
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        Xtr = X[tr].astype(float)
        mu = Xtr.mean(0)
        U = np.linalg.svd(Xtr - mu, full_matrices=False)[2][:k]
        Ztr, Zte = (Xtr - mu) @ U.T, (X[te] - mu) @ U.T
        sd = Ztr.std(0) + 1e-8
        Ztr, Zte = Ztr / sd, Zte / sd
        ytr, yte = y[tr], y[te]
        g_ax = _axis(Ztr, ytr)
        btr, bte = bins[tr], bins[te]
        loc = {b: _axis(Ztr[btr == b], ytr[btr == b]) for b in range(n_bins)}
        rbins = rng.permutation(btr)                        # same bin sizes, direction ignored
        rloc = {b: _axis(Ztr[rbins == b], ytr[rbins == b]) for b in range(n_bins)}
        cos_l = np.array([abs(loc[b] @ g_ax) for b in range(n_bins)])
        res["cos_local_global_per_bin"].append(cos_l.tolist())
        res["cos_local_global"].append(float(cos_l.mean()))
        res["cos_random_global"].append(float(np.mean([abs(rloc[b] @ g_ax) for b in range(n_bins)])))
        res["r2_global"].append(float(_proj_r2(Ztr, ytr, btr, Zte, yte, bte, lambda b: g_ax)))
        res["r2_local"].append(float(_proj_r2(Ztr, ytr, btr, Zte, yte, bte, lambda b: loc[b])))
        # random-bin axes scored on the direction bins' calibration, axis of the random bin a test row would fall in
        # is undefined, so each direction bin b borrows random bin b's axis (a same-size fit without direction info)
        res["r2_random"].append(float(_proj_r2(Ztr, ytr, btr, Zte, yte, bte, lambda b: rloc[b])))
    summ = {q: float(np.mean(v)) for q, v in res.items() if q != "cos_local_global_per_bin"}
    summ.update({q + "_se": float(np.std(v, ddof=1) / np.sqrt(len(v))) for q, v in res.items()
                 if q != "cos_local_global_per_bin"})
    summ["cos_local_global_per_bin_mean"] = np.mean(res["cos_local_global_per_bin"], 0).tolist()
    summ["local_minus_global_r2"] = summ["r2_local"] - summ["r2_global"]
    summ["local_minus_random_r2"] = summ["r2_local"] - summ["r2_random"]
    summ["rotates"] = bool(summ["local_minus_global_r2"] > 0.05 and summ["local_minus_random_r2"] > 0.05
                           and summ["cos_local_global"] < summ["cos_random_global"] - 0.05)
    summ["per_split"] = res
    return summ


def real_code_sd(X, y):
    """SD of the activations along the real (raw-space, ridge) speed direction: the real code's strength."""
    Xc = X - X.mean(0)
    w = _unit(RidgeCV(alphas=ALPHAS).fit(Xc / (Xc.std(0) + 1e-8), y).coef_ / (Xc.std(0) + 1e-8))
    return float((Xc @ w).std())


def plant(X, theta, label, kind, scale, seed=0, ref_sd=None):
    """Add a synthetic code for `label` to X: rotating (axis cos theta u + sin theta v) or constant (axis u).
    Amplitude a = scale * ref_sd per label SD (default ref_sd: the activations' SD along a random direction)."""
    rng = np.random.default_rng(seed)
    D = X.shape[1]
    u, v = _unit(rng.standard_normal(D)), rng.standard_normal(D)
    v = _unit(v - (v @ u) * u)
    r = _unit(rng.standard_normal(D))
    a = scale * (ref_sd if ref_sd is not None else float((X @ r).std()))
    z = (label - label.mean()) / label.std()
    t = np.radians(theta)
    axis = np.cos(t)[:, None] * u + np.sin(t)[:, None] * v if kind == "rotating" else np.broadcast_to(u, X.shape)
    return X + a * z[:, None] * axis, a


def run_layer(d, args):
    tr = d["is_train"]
    X, y, th, fold = d["X"][tr], d["y"][tr], d["df"]["theta_degrees"].to_numpy(float)[tr], d["fold"][tr]
    real = detect(X, y, th, fold, args.n_bins, args.k, args.seed)
    perm = np.random.default_rng(args.seed + 1).permutation(y)
    planted = {}
    ref_sd = real_code_sd(X, y)
    for kind in ("rotating", "constant"):
        Xp, a = plant(X, th, perm, kind, args.plant_scale, args.seed, ref_sd)
        planted[kind] = {"amplitude": a, **detect(Xp, perm, th, fold, args.n_bins, args.k, args.seed)}
    unplanted = detect(X, perm, th, fold, args.n_bins, args.k, args.seed)
    return {"n_train": int(tr.sum()), "real_speed": real, "planted": planted, "real_code_sd": ref_sd,
            "permuted_label_no_plant": {q: unplanted[q] for q in ("r2_global", "r2_local", "r2_random")}}


def verdict_text(L, r):
    s, pr, pc = r["real_speed"], r["planted"]["rotating"], r["planted"]["constant"]
    ok = pr["rotates"] and not pc["rotates"]
    tilt = s["cos_local_global"] < s["cos_random_global"] - 0.05
    return (f"point {L}: speed axis {'ROTATES with direction (cone/other)' if s['rotates'] else 'constant across directions (cylinder)'}"
            f" -- held-out R^2 global {s['r2_global']:.3f} vs local (22.5-deg bins) {s['r2_local']:.3f} vs random-bin "
            f"local {s['r2_random']:.3f}; |cos(local, global)| {s['cos_local_global']:.2f} vs random-bin null "
            f"{s['cos_random_global']:.2f}"
            + (" (local axes tilt away from the global axis more than same-size random bins do, yet predict worse "
               "than it: a shared speed axis plus a smaller direction-dependent component, not a rotating code)"
               if tilt and not s["rotates"] else "")
            + f". Detector check {'passes' if ok else 'FAILS'}: planted rotating code "
            f"R^2 global {pr['r2_global']:.3f} / local {pr['r2_local']:.3f} / random {pr['r2_random']:.3f}, cos "
            f"{pr['cos_local_global']:.2f} vs null {pr['cos_random_global']:.2f} (flagged={pr['rotates']}); planted "
            f"constant code R^2 {pc['r2_global']:.3f} / {pc['r2_local']:.3f} / {pc['r2_random']:.3f}, cos "
            f"{pc['cos_local_global']:.2f} vs {pc['cos_random_global']:.2f} (flagged={pc['rotates']}).")


def plot(res, path):
    layers = sorted(res)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    cases = [("real speed", lambda r: r["real_speed"]), ("planted rotating", lambda r: r["planted"]["rotating"]),
             ("planted constant", lambda r: r["planted"]["constant"])]
    w = 0.13
    xs = np.arange(len(cases))
    cols = {"r2_global": "#666666", "r2_local": "#2a6fdb", "r2_random": "#9bbcf0"}
    for li, L in enumerate(layers):
        for qi, q in enumerate(cols):
            vals = [f(res[L])[q] for _, f in cases]
            err = [f(res[L])[q + "_se"] for _, f in cases]
            off = (li * 3 + qi - 2.5) * w
            axes[0].bar(xs + off, vals, w, yerr=err, color=cols[q], alpha=1.0 if li == 0 else 0.6,
                        label=f"{q[3:]} (point {L})")
        for i, (_, f) in enumerate(cases):
            r = f(res[L])
            axes[1].scatter([r["cos_random_global"]], [r["cos_local_global"]], marker="o" if li == 0 else "s",
                            color=["#2a6fdb", "#c0392b", "#27ae60"][i], label=f"{cases[i][0]} (point {L})")
    axes[0].set_xticks(xs, [c[0] for c in cases])
    axes[0].set_ylabel("held-out R^2 (1-D projection)")
    axes[0].legend(fontsize=6, frameon=False, ncol=2)
    lim = [0, 1]
    axes[1].plot(lim, lim, color="#aaaaaa", lw=0.8)
    axes[1].set_xlabel("|cos(random-bin axis, global)| (null)")
    axes[1].set_ylabel("|cos(direction-bin axis, global)|")
    axes[1].legend(fontsize=6, frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Rotating-code detector on the speed set: local vs global speed axis", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(args):
    rdir, fdir = Path(args.results_dir), Path(args.figures_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    out = {"dataset": "speed", "variable": "speed", "n_bins": args.n_bins, "k": args.k, "plant_scale": args.plant_scale,
           "provenance": provenance(args.split, seeds={"random_bins_and_plants": args.seed, "label_permutation":
                                                       args.seed + 1}, layers=args.layers, k=args.k,
                                    n_bins=args.n_bins, plant_scale=args.plant_scale),
           "rows": "train folds 0-4 only, 5-fold CV over them; test clips unread",
           "rotates_rule": "local - global R^2 > 0.05 and local - random-bin R^2 > 0.05 and cos(local, global) < "
                           "random-bin null - 0.05",
           "layers": {}}
    res = {}
    for L in args.layers:
        d = load_inputs("speed", L, "speed", args.act_dir, args.table, args.split)
        r = run_layer(d, args)
        r["layer_role"] = layer_role("speed", L)["layer_role"]
        r["verdict"] = verdict_text(L, r)
        res[L] = r
        out["layers"][str(L)] = r
        print(r["verdict"], flush=True)
    out["verdict"] = " ".join(out["layers"][str(L)]["verdict"] for L in args.layers)
    tag = "_".join(f"L{L}" for L in args.layers)
    (rdir / f"p2_rotating_speed_axis_{tag}.json").write_text(json.dumps(out, indent=1))
    plot(res, fdir / f"fig4f_rotating_speed_axis_{tag}.png")
    return out


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, nargs="+", default=[12, 22])
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--n-bins", type=int, default=16)
    p.add_argument("--plant-scale", type=float, default=1.0,
                   help="planted amplitude per label SD, in units of the activations' SD along the real speed direction")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--act-dir", default=None)
    p.add_argument("--table", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--figures-dir", default=str(PROJECT_ROOT / "figures"))
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse())

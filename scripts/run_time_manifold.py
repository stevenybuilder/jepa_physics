"""Experiment A (Part 2 new variable): within-clip elapsed TIME as a manifold variable, clock vs odometer.

Features: timepool [N, 26, 8, 1024] (per tubelet step, 2 frames each; step time tau_t = (2t + 0.5) / 24 s) and
diskpool (same, pooled over the disk's patches) as a check. Roles from wm.p2_data.load_inputs + splits/split_v1.json:
knot folds 0-2 build PCA / centroids / splines, probe folds 3-4 fit readouts, test clips are steered.

 1 decode : ridge probes (grouped by the split's 5 train folds, all 8 steps of a clip in one fold) for step t,
            distance travelled s = v tau + a tau^2 / 2 ("odometer"), and absolute position (x, y); CV R^2 / MAE on
            train folds, and R^2 / MAE on test clips from a fit on all train clips. Also the t-probe's signed bias
            per speed (acceleration) band: a clock reads t the same at every speed, an odometer reads fast clips late.
 2 geometry: residual = step feature - the clip's own mean over its 8 steps; PCA-64 on knot residual rows;
            per-step centroids; fraction of held-out per-step variance explained by the centroids, a count-weighted
            smoothing spline over t, and a straight line in t; centroid participation ratio; interior bend /
            centroid noise (spread_vs_curvature) and leave-one-step-out sagitta / centroid noise.
 3 clock vs odometer: a time spline (linearly continued past t = 0, 7) from MID-band knot clips only; each
            held-out clip's step-t residual gets a coordinate (nearest point of the spline); per-clip slope of the
            coordinate on t, per band, against the clock (1) and odometer (v / v_mid; sqrt(a / a_mid) for the
            acceleration set, from rest) predictions; plus coord = a + b t + c odo over all held-out clips.
            A mid-band linear time probe (ridge on the residual PCA coordinates) is the second coordinate.
 4 steering (speed set): hold out step h (2..5) from PCA, centroids and spline; move a test clip's step-t feature
            to h along the spline vs along the raw-centroid chord (the A.9 chord between the neighbouring raw
            centroids) vs unedited, additive in PCA-64 with the residual kept. Read with disjoint probes fit on the
            probe clips' per-step features (t, distance, (x, y), speed, direction) and nearest-real agreement with
            the clip's own real step-h feature. Gaps spline - chord with a clip bootstrap.

Writes results/p5_time_manifold.json and figures/fig_time_manifold.png.

  python scripts/run_time_manifold.py --jobs 12
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))  # run without installing wm
from wm import geometry_checks as gc
from wm import manifold as mf
from wm import timeman as tm
from wm.data import PROJECT_ROOT
from wm.p2_data import load_inputs
from wm.provenance import provenance

T = tm.N_STEPS
BANDS = {"speed": {"var": "speed_mps", "slow": (0.0, 1.0), "mid": (1.5, 2.75), "fast": (3.25, 9.0)},
         "acceleration": {"var": "acceleration_mps2", "slow": (0.0, 2.5), "mid": (4.0, 6.5), "fast": (7.75, 99.0)},
         "direction": {"var": "speed_mps", "slow": (0.0, 2.0), "mid": (3.0, 5.0), "fast": (6.0, 99.0)}}


def meta(dataset):
    d = load_inputs(dataset, 0)
    df = d["df"]
    v, a = df["speed_mps"].to_numpy(float), df["acceleration_mps2"].to_numpy(float)
    return {"df": df, "role": d["role"], "fold": d["fold"], "train": d["is_train"], "ids": df["id"].to_numpy(),
            "dist": tm.step_distance(v, a), "speed": v, "accel": a, "theta": df["theta_degrees"].to_numpy(float),
            "pos": tm.step_position(df[["start_x", "start_y"]].to_numpy(float), df["theta_degrees"], v, a)}


def band_of(m, dataset):
    b = BANDS[dataset]
    x = m["df"][b["var"]].to_numpy(float)
    out = np.full(len(x), "", dtype=object)
    for k in ("slow", "mid", "fast"):
        lo, hi = b[k]
        out[(x >= lo) & (x <= hi)] = k
    if dataset == "direction":                       # accelerating clips (speed 0) are not in any speed band
        out[m["df"]["motion"].to_numpy() == "acceleration"] = ""
    return out


def load_layer(path, layer):
    """[N, 8, D] float32 at `layer` from {pool}.npy [N, 26, 8, D], or from a per-layer slice {pool}_L{layer}.npy."""
    path = Path(path)
    if path.exists():
        F = np.asarray(np.load(path, mmap_mode="r")[:, layer], dtype=np.float32)
    else:
        F = np.load(path.with_name(f"{path.stem}_L{layer}.npy")).astype(np.float32)
    bad = ~np.isfinite(F).all(-1)
    return F, bad


# ---------------------------------------------------------------- 1. decodability ---------------------------------

def decode_layer(path, layer, dataset):
    m = meta(dataset)
    F, bad = load_layer(path, layer)
    N = len(F)
    t = np.broadcast_to(np.arange(T), (N, T))
    Y = np.stack([t, m["dist"], m["pos"][..., 0], m["pos"][..., 1]], -1).reshape(N * T, 4)
    X = F.reshape(N * T, -1)
    ok = ~bad.reshape(-1)
    clip = np.repeat(np.arange(N), T)
    tr, te = np.repeat(m["train"], T) & ok, np.repeat(~m["train"], T) & ok
    fold = np.repeat(m["fold"], T)
    fit = lambda A, B: tm.make_pipeline(tm.StandardScaler(), tm.RidgeCV(alphas=tm.ALPHAS, alpha_per_target=True)).fit(A, B)
    oof = tm.grouped_cv_predict(X[tr], Y[tr], fold[tr], fit)
    pte = fit(X[tr], Y[tr]).predict(X[te])

    def scores(y, p):
        return {"t_r2": tm.r2(y[:, 0], p[:, 0]), "t_mae_steps": float(np.abs(y[:, 0] - p[:, 0]).mean()),
                "dist_r2": tm.r2(y[:, 1], p[:, 1]), "dist_mae_m": float(np.abs(y[:, 1] - p[:, 1]).mean()),
                "xy_r2": tm.r2(y[:, 2:], p[:, 2:]), "xy_err_m": float(np.linalg.norm(y[:, 2:] - p[:, 2:], axis=1).mean())}
    out = {"cv": scores(Y[tr], oof), "test": scores(Y[te], pte), "n_rows_dropped_nonfinite": int((~ok).sum())}
    band = np.repeat(band_of(m, dataset), T)[tr]
    out["cv_t_bias_by_band"] = {k: float((oof[band == k, 0] - Y[tr][band == k, 0]).mean()) for k in ("slow", "mid", "fast")}
    # the bias of an odometer-driven readout: fast clips at step t look like mid clips at a later step
    out["cv_t_bias_by_band_per_step"] = {k: [float((oof[(band == k) & (Y[tr][:, 0] == s), 0] - s).mean())
                                             for s in range(T)] for k in ("slow", "mid", "fast")}
    return layer, out


# ---------------------------------------------------------------- 2. geometry -------------------------------------

def fit_time_geometry(R, m, rows, k=64, exclude_step=None):
    """PCA-k on the residual rows of clips `rows` (all steps but exclude_step), per-step centroids, smoothing
    spline and interpolating spline over t. Returns dict(pca, cent, spline, interp, Z [N, T, k])."""
    steps = np.array([s for s in range(T) if s != exclude_step])
    Rk = R[rows][:, steps].reshape(-1, R.shape[-1])
    pca = mf.fit_pca(Rk, k)
    Z = pca.project(R)
    lab = np.broadcast_to(steps, (rows.sum(), len(steps))).reshape(-1)
    cent = mf.centroids(Z[rows][:, steps].reshape(-1, Z.shape[-1]), lab)
    return {"pca": pca, "cent": cent, "Z": Z, "steps": steps,
            "spline": mf.fit_curve(cent, False, spline="smooth", extend="linear"),
            "interp": mf.fit_curve(cent, False, spline="interp")}


def geometry_layer(path, layer, dataset, k=64):
    m = meta(dataset)
    F, _ = load_layer(path, layer)
    F = np.nan_to_num(F)
    R = tm.remove_clip_mean(F)
    knot, held = m["role"] == "knot", np.isin(m["role"], ["probe", "test"])
    g = fit_time_geometry(R, m, knot, k)
    Z, C, pca = g["Z"], g["cent"]["C"], g["pca"]
    Zh = Z[held].reshape(-1, Z.shape[-1])
    th = np.broadcast_to(np.arange(T), (held.sum(), T)).reshape(-1)
    S = g["spline"](np.arange(T, dtype=float))
    A = np.stack([np.ones(T), np.arange(T)], 1)
    L = A @ np.linalg.lstsq(A, C, rcond=None)[0]
    Rh = R[held].reshape(-1, R.shape[-1])
    full_res = ((Rh - pca.lift_delta(C[th])) ** 2).sum()
    per_clip = 1 - ((Z[held] - C[None]) ** 2).sum((1, 2)) / (Z[held] ** 2).sum((1, 2))
    tot_var = float(((F[held] - F[held].mean((0, 1))) ** 2).sum())
    cent_noise = g["cent"]["spread"] / np.sqrt(g["cent"]["count"])
    svc = gc.spread_vs_curvature(g["interp"], g["cent"]["spread"], g["cent"]["count"])
    lab_sv = np.linalg.svd(C - C.mean(0), compute_uv=False) ** 2
    loo = []
    for h in range(1, T - 1):
        gh = fit_time_geometry(R, m, knot, k, exclude_step=h)
        Ch, sp = gh["cent"]["C"], gh["spline"]
        i = list(gh["steps"]).index(h - 1)
        chord = (Ch[i] + Ch[i + 1]) / 2
        truth = gh["Z"][knot][:, h].mean(0)
        noise = float(np.median(gh["cent"]["spread"] / np.sqrt(gh["cent"]["count"])))
        loo.append({"held_step": h, "sagitta": float(np.linalg.norm(sp(float(h)) - chord)),
                    "sagitta_over_centroid_noise": float(np.linalg.norm(sp(float(h)) - chord) / noise),
                    "spline_err_to_heldout_centroid": float(np.linalg.norm(sp(float(h)) - truth)),
                    "chord_err_to_heldout_centroid": float(np.linalg.norm(chord - truth))})
    return layer, {
        "residual_var_frac_of_step_var": float((R[held] ** 2).sum() / tot_var),
        "heldout_var_explained": {"centroids": tm.step_variance_explained(Zh, th, C),
                                  "spline": tm.step_variance_explained(Zh, th, S),
                                  "line_in_t": tm.step_variance_explained(Zh, th, L),
                                  "centroids_full_space": float(1 - full_res / (Rh ** 2).sum())},
        "per_clip_var_explained_median": float(np.median(per_clip)),
        "centroid_participation_ratio": gc.participation_ratio(C),
        "centroid_pc1_frac": float(lab_sv[0] / lab_sv.sum()),
        "centroid_path_length": float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum()),
        "centroid_end_to_end": float(np.linalg.norm(C[-1] - C[0])),
        "median_centroid_noise": float(np.median(cent_noise)),
        "bend": svc,
        "loo_sagitta": loo,
        "loo_sagitta_over_noise_median": float(np.median([r["sagitta_over_centroid_noise"] for r in loo])),
        "loo_spline_beats_chord": int(sum(r["spline_err_to_heldout_centroid"] < r["chord_err_to_heldout_centroid"] for r in loo)),
        "centroids_top2": ((C - C.mean(0)) @ np.linalg.svd(C - C.mean(0), full_matrices=False)[2][:2].T).tolist(),
    }


# ---------------------------------------------------------------- 3. clock vs odometer ----------------------------

def clock_layer(path, layer, dataset, k=64, n_boot=500):
    m = meta(dataset)
    F, _ = load_layer(path, layer)
    R = tm.remove_clip_mean(np.nan_to_num(F))
    band = band_of(m, dataset)
    knot = m["role"] == "knot"
    held = ~knot & (band != "")
    mid_knot = knot & (band == "mid")
    g = fit_time_geometry(R, m, knot, k)                 # PCA on all knot rows (label-free), spline on mid band only
    Z = g["Z"]
    cent = mf.centroids(Z[mid_knot].reshape(-1, Z.shape[-1]), np.broadcast_to(np.arange(T), (mid_knot.sum(), T)).reshape(-1))
    curve = mf.fit_curve(cent, False, spline="smooth", extend="linear")
    gt = np.linspace(-6, 20, 5201)
    coord_sp = tm.nearest_coord(Z, gt, curve(gt))
    probe = tm.ridge(Z[mid_knot].reshape(-1, Z.shape[-1]), np.broadcast_to(np.arange(T), (mid_knot.sum(), T)).reshape(-1))
    coord_lin = probe.predict(Z.reshape(-1, Z.shape[-1])).reshape(len(Z), T)
    if dataset == "acceleration":
        ref = float(m["accel"][mid_knot].mean())
        odo = tm.odometer_coord(m["dist"], ref_accel=ref)
    else:
        ref = float(m["speed"][mid_knot].mean())
        odo = tm.odometer_coord(m["dist"], ref_speed=ref)
    tgrid = np.broadcast_to(np.arange(T, dtype=float), odo.shape)
    out = {"reference_mid_mean": ref, "n_heldout_clips": {b: int((held & (band == b)).sum()) for b in ("slow", "mid", "fast")}}
    rng = np.random.default_rng(0)
    for name, coord in (("spline", coord_sp), ("linear_probe", coord_lin)):
        slope = tm.per_clip_slope(coord)
        odo_slope = tm.per_clip_slope(odo)
        res = {}
        for b in ("slow", "mid", "fast"):
            r = held & (band == b)
            res[b] = {"slope": tm.cluster_bootstrap(slope[r], n_boot=n_boot),
                      "clock_prediction": 1.0, "odometer_prediction": float(odo_slope[r].mean()),
                      "mean_coord_per_step": coord[r].mean(0).tolist()}
        s_slow, s_fast = slope[held & (band == "slow")], slope[held & (band == "fast")]
        boots = [rng.choice(s_fast, len(s_fast)).mean() / rng.choice(s_slow, len(s_slow)).mean() for _ in range(n_boot)]
        res["fast_over_slow_slope"] = {"mean": float(s_fast.mean() / s_slow.mean()),
                                       "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
                                       "clock_prediction": 1.0,
                                       "odometer_prediction": float(odo_slope[held & (band == "fast")].mean()
                                                                    / odo_slope[held & (band == "slow")].mean())}
        idx = np.flatnonzero(held)
        joint = tm.clock_odometer_fit(coord[idx], tgrid[idx], odo[idx])
        bs = [tm.clock_odometer_fit(coord[j], tgrid[j], odo[j]) for j in
              (rng.choice(idx, len(idx)) for _ in range(200))]
        for key in ("b_clock", "c_odometer"):
            joint[key + "_ci95"] = [float(x) for x in np.percentile([b[key] for b in bs], [2.5, 97.5])]
        res["joint_fit_heldout"] = joint
        # curvature per band (quadratic term of coord on t, per clip): odometer on the acceleration set with a
        # t-indexed reference is still linear in tau, so this is a check, not the test
        tc = np.arange(T) - (T - 1) / 2
        Q = np.stack([np.ones(T), tc, tc ** 2 - (tc ** 2).mean()], 1)
        quad = np.linalg.lstsq(Q, coord.T, rcond=None)[0][2]
        res["quadratic_coef_by_band"] = {b: tm.cluster_bootstrap(quad[held & (band == b)], n_boot=n_boot) for b in ("slow", "mid", "fast")}
        out[name] = res
    return layer, out


# ---------------------------------------------------------------- 4. steering -------------------------------------

def steer_layer(path, layer, dataset="speed", k=64, held_steps=(2, 3, 4, 5), n_boot=1000):
    m = meta(dataset)
    F, _ = load_layer(path, layer)
    F = np.nan_to_num(F).astype(np.float64)
    R = tm.remove_clip_mean(F)
    knot, probe, test = (m["role"] == r for r in ("knot", "probe", "test"))
    N, D = F.shape[0], F.shape[-1]
    th = np.radians(m["theta"])
    Yp = np.column_stack([np.broadcast_to(np.arange(T), (N, T)).reshape(-1), m["dist"].reshape(-1),
                          m["pos"].reshape(-1, 2), np.repeat(m["speed"], T),
                          np.repeat(np.sin(th), T), np.repeat(np.cos(th), T)])
    pr = np.repeat(probe, T)
    readout = tm.make_pipeline(tm.StandardScaler(), tm.RidgeCV(alphas=tm.ALPHAS, alpha_per_target=True)).fit(
        F.reshape(-1, D)[pr], Yp[pr])

    def read(X):
        p = readout.predict(X)
        return {"t": p[:, 0], "dist": p[:, 1], "xy": p[:, 2:4], "speed": p[:, 4],
                "dir": np.degrees(np.arctan2(p[:, 5], p[:, 6])) % 360}

    rows = {q: [] for q in ("arm", "clip", "h", "src", "time_err", "dist_err", "xy_err", "speed_change", "dir_change",
                             "nearest_real_R", "edit_norm", "band")}
    band = band_of(m, dataset)
    tidx = np.flatnonzero(test)
    for h in held_steps:
        g = fit_time_geometry(R, m, knot, k, exclude_step=h)
        pca, sp = g["pca"], g["spline"]
        steps, Craw = g["steps"], g["cent"]["C"]
        raw_at = lambda tt: np.stack([np.interp(tt, steps, Craw[:, j]) for j in range(Craw.shape[1])], -1)
        for s in steps:
            X0 = F[tidx, s]
            Xh = F[tidx, h]
            dz = {"spline": sp(float(h)) - sp(float(s)), "chord_raw": raw_at(float(h)) - raw_at(float(s))}
            base = read(X0)
            for arm in ("unedited", "spline", "chord_raw", "donor_real_step"):
                if arm == "unedited":
                    Xs = X0
                elif arm == "donor_real_step":
                    Xs = Xh
                else:
                    Xs = X0 + pca.lift_delta(dz[arm])[None]
                r = read(Xs)
                n = len(tidx)
                rows["arm"] += [arm] * n
                rows["clip"].append(m["ids"][tidx]); rows["h"].append(np.full(n, h)); rows["src"].append(np.full(n, s))
                rows["time_err"].append(np.abs(r["t"] - h))
                rows["dist_err"].append(np.abs(r["dist"] - m["dist"][tidx, h]))
                rows["xy_err"].append(np.linalg.norm(r["xy"] - m["pos"][tidx, h], axis=1))
                rows["speed_change"].append(np.abs(r["speed"] - base["speed"]))
                rows["dir_change"].append(mf.value_error(r["dir"], base["dir"], True))
                rows["nearest_real_R"].append(mf.nearest_real_agreement(Xs, X0, Xh) if arm != "unedited" else np.zeros(n))
                rows["edit_norm"].append(np.linalg.norm(Xs - X0, axis=1))
                rows["band"].append(band[tidx])
    arm = np.array(rows.pop("arm"))
    cat = {q: np.concatenate(v) for q, v in rows.items()}
    metrics = ("time_err", "dist_err", "xy_err", "speed_change", "dir_change", "nearest_real_R", "edit_norm")
    out = {"n_steers_per_arm": int((arm == "spline").sum()), "held_steps": list(held_steps), "summary": {}, "gaps": {}}
    for a in ("unedited", "spline", "chord_raw", "donor_real_step"):
        out["summary"][a] = {q: float(cat[q][arm == a].mean()) for q in metrics}
    for other in ("chord_raw", "unedited"):
        out["gaps"][f"spline_minus_{other}"] = {
            q: tm.cluster_bootstrap(cat[q][arm == "spline"] - cat[q][arm == other], cat["clip"][arm == "spline"], n_boot=n_boot)
            for q in metrics if not (other == "unedited" and q in ("nearest_real_R", "edit_norm"))}
    out["dist_err_by_band"] = {a: {b: float(cat["dist_err"][(arm == a) & (cat["band"] == b)].mean())
                                   for b in ("slow", "mid", "fast")} for a in ("spline", "chord_raw", "donor_real_step")}
    out["time_err_by_band"] = {a: {b: float(cat["time_err"][(arm == a) & (cat["band"] == b)].mean())
                                   for b in ("slow", "mid", "fast")} for a in ("spline", "chord_raw", "donor_real_step")}
    sh = np.abs(cat["h"] - cat["src"])
    out["time_err_by_shift"] = {a: {str(d): float(cat["time_err"][(arm == a) & (sh == d)].mean()) for d in range(1, T)
                                    if ((arm == a) & (sh == d)).any()} for a in ("spline", "chord_raw")}
    return layer, out


# ---------------------------------------------------------------- driver ------------------------------------------

def act(dataset, model, pool, root):
    sub = {"vjepa2": "vjepa2", "random": "random"}[model]
    p = Path(root) / dataset / sub / f"{pool}.npy"
    if model == "random" and not p.exists():
        p = Path(root) / dataset / "random_tp" / f"{pool}.npy"
    return p


def available_layers(path, wanted):
    """All `wanted` points if the full array exists, else the points that have a per-layer slice."""
    path = Path(path)
    if path.exists():
        return list(wanted)
    return [l for l in wanted if path.with_name(f"{path.stem}_L{l}.npy").exists()]


def figure(res, path):
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    a = ax[0, 0]
    for key, st in (("speed/vjepa2/timepool", "-"), ("direction/vjepa2/timepool", "-"), ("direction/random/timepool", "--"),
                    ("speed/vjepa2/diskpool", ":"), ("acceleration/vjepa2/timepool", "-")):
        if key in res["decode"]:
            d = res["decode"][key]
            L = sorted(int(x) for x in d)
            a.plot(L, [d[str(l)]["cv"]["t_r2"] for l in L], st, label=f"t  {key.replace('/timepool', '')}")
    d = res["decode"].get("speed/vjepa2/timepool")
    if d:
        L = sorted(int(x) for x in d)
        a.plot(L, [d[str(l)]["cv"]["dist_r2"] for l in L], "k-.", label="distance, speed set")
    a.set_xlabel("point (block)"); a.set_ylabel("grouped-CV R²"); a.set_title("Decoding elapsed step t (and distance)")
    a.legend(fontsize=7); a.set_ylim(0, 1.02)

    a = ax[0, 1]
    g = res["geometry"]["speed/vjepa2/timepool"].get("12")
    if g:
        P = np.array(g["centroids_top2"])
        a.plot(P[:, 0], P[:, 1], "o-")
        for i, p in enumerate(P):
            a.annotate(str(i), p, fontsize=8)
        a.set_title(f"Point 12, speed set: step centroids after removing each clip's mean\n"
                    f"held-out variance explained: centroids {g['heldout_var_explained']['centroids']:.2f}, "
                    f"line {g['heldout_var_explained']['line_in_t']:.2f}", fontsize=9)
        a.set_xlabel("PC1 of step centroids"); a.set_ylabel("PC2")

    a = ax[1, 0]
    for ds, mk in (("speed", "o"), ("acceleration", "s")):
        c = res["clock"].get(f"{ds}/vjepa2/timepool", {})
        L = sorted(int(x) for x in c)
        for b, col in (("slow", "tab:blue"), ("mid", "tab:gray"), ("fast", "tab:red")):
            y = [c[str(l)]["spline"][b]["slope"]["mean"] for l in L]
            lo = [c[str(l)]["spline"][b]["slope"]["ci95"][0] for l in L]
            hi = [c[str(l)]["spline"][b]["slope"]["ci95"][1] for l in L]
            a.errorbar(np.array(L) + (0.15 if ds == "acceleration" else 0), y, yerr=[np.subtract(y, lo), np.subtract(hi, y)],
                       fmt=mk + ("-" if ds == "speed" else "--"), color=col, ms=4, label=f"{ds} {b}")
            if L:
                a.axhline(c[str(L[0])]["spline"][b]["odometer_prediction"], color=col, lw=0.6,
                          ls="-" if ds == "speed" else "--")
    a.axhline(1.0, color="k", lw=1.2)
    a.set_xlabel("point"); a.set_ylabel("slope of spline coordinate on t")
    a.set_title("Clock (black line = 1) vs odometer (thin lines) per band", fontsize=9); a.legend(fontsize=6, ncol=2)

    a = ax[1, 1]
    s = res["steer"]
    L = sorted(int(x) for x in s)
    w = 0.2
    for i, arm in enumerate(("unedited", "chord_raw", "spline", "donor_real_step")):
        a.bar(np.arange(len(L)) + (i - 1.5) * w, [s[str(l)]["summary"][arm]["time_err"] for l in L], w, label=arm)
    a.set_xticks(range(len(L))); a.set_xticklabels([f"pt {l}" for l in L])
    a.set_ylabel("time-probe error to target step (steps)"); a.legend(fontsize=7)
    a.set_title("Steering a step-t feature to held-out step h (speed set test clips)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-root", default=str(PROJECT_ROOT / "artifacts" / "activations"))
    ap.add_argument("--decode-layers", type=int, nargs="+", default=list(range(26)))
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 12, 19, 22])
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "p5_time_manifold.json"))
    ap.add_argument("--fig", default=str(PROJECT_ROOT / "figures" / "fig_time_manifold.png"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--skip-decode", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    res = {"decode": {}, "geometry": {}, "clock": {}, "steer": {}}
    decode_cfgs = [("direction", "vjepa2", "timepool"), ("direction", "vjepa2", "diskpool"), ("direction", "random", "timepool"),
                   ("speed", "vjepa2", "timepool"), ("speed", "vjepa2", "diskpool"),
                   ("acceleration", "vjepa2", "timepool"), ("acceleration", "vjepa2", "diskpool")]
    geom_cfgs = [("speed", "vjepa2", "timepool"), ("speed", "vjepa2", "diskpool"), ("direction", "vjepa2", "timepool"),
                 ("direction", "random", "timepool"), ("acceleration", "vjepa2", "timepool")]
    clock_cfgs = [("speed", "vjepa2", "timepool"), ("speed", "vjepa2", "diskpool"), ("acceleration", "vjepa2", "timepool"),
                  ("acceleration", "vjepa2", "diskpool"), ("direction", "vjepa2", "timepool"), ("direction", "random", "timepool")]
    jobs = []
    if not args.skip_decode:
        jobs += [("decode", c, delayed(decode_layer)(act(*c, args.act_root), l, c[0])) for c in decode_cfgs
                 for l in available_layers(act(*c, args.act_root), args.decode_layers)]
    jobs += [("geometry", c, delayed(geometry_layer)(act(*c, args.act_root), l, c[0])) for c in geom_cfgs for l in args.layers]
    jobs += [("clock", c, delayed(clock_layer)(act(*c, args.act_root), l, c[0])) for c in clock_cfgs for l in args.layers]
    jobs += [("steer", ("speed", "vjepa2", "timepool"), delayed(steer_layer)(act("speed", "vjepa2", "timepool", args.act_root), l))
             for l in args.layers]
    outs = Parallel(n_jobs=args.jobs, verbose=5)(j[2] for j in jobs)
    for (sec, c, _), (layer, o) in zip(jobs, outs):
        if sec == "steer":
            res["steer"][str(layer)] = o
        else:
            res[sec].setdefault("/".join(c), {})[str(layer)] = o
    res["keys"] = {
        "decode[dataset/model/pool][point]": "cv = grouped CV over the split's 5 train folds (all 8 steps of a clip together); test = fit on all train clips, read on test clips. t_r2/t_mae_steps: step index 0..7; dist: distance travelled from the start (m, v tau + a tau^2/2); xy: absolute disk position at the step (m). cv_t_bias_by_band: mean signed error of the t probe per speed band (acceleration band on the acceleration set; speed band on the direction set, whose accelerating clips have speed 0 and fall in 'slow').",
        "geometry[...][point]": "residual = step feature - clip mean over 8 steps; PCA-64 on knot residual rows; heldout_var_explained: fraction of non-knot clips' residual variance (PCA-64) explained by the knot step centroids / smoothing spline at integer t / least-squares line in t (centroids_full_space: centroids lifted, over full residual variance); residual_var_frac_of_step_var: within-clip (time) share of all per-step variance; bend: spread_vs_curvature on the interpolating spline (interior knots); loo_sagitta: step h held out of PCA+centroids+spline, sagitta = |spline(h) - raw chord midpoint|, over median centroid noise (spread/sqrt(count)); loo_spline_beats_chord = # of the 6 held-out steps whose true centroid the spline is closer to than the chord.",
        "clock[...][point]": "time spline from MID-band knot clips (extended linearly), coordinate of each non-knot clip's step residual = nearest spline point; also a mid-band ridge time probe on residual PCA coords (linear_probe). slope: per-clip OLS slope of coord on t, mean + clip bootstrap CI; clock_prediction 1, odometer_prediction: mean per-clip slope of the odometer coordinate (the step at which a mid-mean reference clip has covered the same distance; speed v/v_mid, acceleration sqrt(a/a_mid)). joint_fit_heldout: coord = a + b t + c odo over held-out rows (clip bootstrap CIs). bands: " + json.dumps(BANDS),
        "steer[point]": "speed set, held-out step h in 2..5 (excluded from PCA, centroids, spline); every other step s of every test clip is moved to h. Arms: spline (x + lift(S(h) - S(s))), chord_raw (raw-centroid polyline point at h - raw centroid at s; the A.9 chord between the neighbours of h), unedited, donor_real_step (the clip's own real step-h feature = ceiling). Readouts: one ridge fit on probe-fold per-step features for t, distance, (x, y), speed, (sin, cos) direction. time_err = |t_hat - h| (steps); dist_err / xy_err vs the clip's true step-h distance / position (m); speed_change / dir_change = off-target change vs the unedited read; nearest_real_R = 1 - |x' - x_h| / |x_s - x_h| with x_h the clip's own real step-h feature. gaps: paired differences, mean with 95% clip-bootstrap CI.",
    }
    res["provenance"] = provenance(seeds={"bootstrap": 0}, act_root=args.act_root, k=64, decode_layers=args.decode_layers,
                                   layers=args.layers, step_time="tau_t = (2t + 0.5)/24 s",
                                   random_init_note="random-init timepool exists only for the direction set (artifacts/activations/direction/random*/timepool.npy); speed/acceleration random-init have meanpool only",
                                   wall_s=round(time.time() - t0, 1))
    Path(args.out).write_text(json.dumps(res, indent=1))
    figure(res, args.fig)
    print("wrote", args.out, args.fig, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

"""Experiment B (Part 2 new variable): the joint VELOCITY SHEET over (direction, speed) on the speed set.

Same roles as run_part2 / run_position_sheet (wm.p2_data.load_inputs, splits/split_v1.json): knot folds 0-2 build
PCA-64, centroids and every curve / sheet; probe folds 3-4 fit the readouts; test clips are steered. Meanpool at one
point. Cells: 16 direction bins (22.5 deg, 4 theta values each) x 8 speed bins (8 speed values each, 0.47 m/s).

Geometry (knot cells vs held-out non-knot cells): how much of the held-out cell-centroid variance each model explains,
  cylinder  C = m(speed bin) + u(dir bin)             (one ring for all speeds)
  cone      C = m(speed bin) + r(speed bin) u(dir bin) (ring size per band; a true cone has r proportional to speed)
  sheet     thin-plate spline over (cos theta, sin theta, lambda * speed), smoothing and lambda chosen on train rows
plus ring radius per speed bin (noise-corrected) and a line r = a + b v (a cone needs a ~ 0 and r(fast)/r(slow) ~ the
speed ratio).

Steering (per seed one held-out block: 2 direction bins x 2 interior speed bins, excluded from PCA, centroids, sheet,
rings and the selection of smoothing): steer test clips to the held-out cells, all arms additive in PCA-64 with the
clip's residual kept, K waypoints:
 (i)  direction at fixed speed (carriers from the target's speed bin, outside the held direction bins):
      sheet (TPS walked along theta at the clip's own speed) | ring1d (periodic smoothing spline over the knot
      direction-bin centroids of that speed bin) | chord_sheet (endpoint-matched straight line) | chord_raw (straight
      line between the raw-centroid polyline points of the band ring, the A.9 chord).
 (ii) speed at fixed direction (carriers from the target's direction bin, outside the held speed bins):
      sheet (TPS walked along speed at the clip's own theta) | speedline (natural smoothing spline over the knot
      speed-value centroids pooled over all directions, the Part 2 1-D speed curve) | chord_sheet | chord_raw.
Readouts fit on probe clips: direction (sin, cos) and speed ridge probes; endpoint error, off-target change of the
other variable, nearest-real agreement with the test clips in the target cell, min direction-probe radius along the
path (direction steers). Gaps sheet - other arm, 95% clip bootstrap.

Writes results/p5_velocity_sheet.json and figures/fig_velocity_sheet.png.

  python scripts/run_velocity_sheet.py --layers 12 19 22
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
from scipy.interpolate import RBFInterpolator

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))  # run without installing wm
from wm import manifold as mf
from wm import timeman as tm
from wm.data import PROJECT_ROOT
from wm.p2_data import load_inputs
from wm.provenance import provenance

N_DIR, N_SPD = 16, 8
V_MID, V_HALF = 2.125, 1.875
LAMBDAS = (0.5, 1.0, 2.0)
SMOOTHINGS = (0.0, 0.1, 1.0, 10.0)
BLOCKS = [(1, (2, 3)), (5, (3, 4)), (9, (4, 5)), (13, (3, 4))]      # (first held dir bin, held speed bins) per seed


def bins(theta, speed):
    sv = np.sort(np.unique(speed))
    sbin = np.searchsorted(sv, speed) * N_SPD // len(sv)
    return tm.angular_bins(theta, N_DIR), sbin


def sheet_inputs(theta_deg, speed, lam):
    th = np.radians(np.asarray(theta_deg, float))
    return np.stack([np.cos(th), np.sin(th), lam * (np.asarray(speed, float) - V_MID) / V_HALF], -1)


def cell_table(Z, theta, speed, dbin, sbin, rows):
    """Per occupied (dir bin, speed bin) cell among `rows`: centroid, mean theta (circular), mean speed, count."""
    out = {"C": [], "theta": [], "speed": [], "count": [], "d": [], "s": [], "se": []}
    for dd in range(N_DIR):
        for ss in range(N_SPD):
            r = rows & (dbin == dd) & (sbin == ss)
            if r.sum() < 2:
                continue
            c = Z[r].mean(0)
            out["C"].append(c)
            out["theta"].append(np.degrees(np.angle(np.exp(1j * np.radians(theta[r])).mean())) % 360)
            out["speed"].append(speed[r].mean()); out["count"].append(int(r.sum())); out["d"].append(dd); out["s"].append(ss)
            out["se"].append(np.sqrt(((Z[r] - c) ** 2).sum(1).mean() / r.sum()))
    return {k: np.array(v) for k, v in out.items()}


def fit_sheet(cells, lam, smoothing):
    return RBFInterpolator(sheet_inputs(cells["theta"], cells["speed"], lam), cells["C"], kernel="thin_plate_spline",
                           smoothing=smoothing)


def choose_sheet(Z, theta, speed, dbin, sbin, rows, eval_block):
    """lambda and smoothing by held-out-block reconstruction on `rows` (train clips), over the other seeds' blocks
    with the evaluation block's cells removed first. Returns (lam, smoothing, table)."""
    ev = np.isin(dbin, [eval_block[0], eval_block[0] + 1]) & np.isin(sbin, eval_block[1])
    rows = rows & ~ev
    table = {}
    for lam in LAMBDAS:
        for sm in SMOOTHINGS:
            errs = []
            for b0, sp in BLOCKS:
                if (b0, sp) == eval_block:
                    continue
                held = np.isin(dbin, [b0, b0 + 1]) & np.isin(sbin, sp)
                kc = cell_table(Z, theta, speed, dbin, sbin, rows & ~held)
                hc = cell_table(Z, theta, speed, dbin, sbin, rows & held)
                pred = fit_sheet(kc, lam, sm)(sheet_inputs(hc["theta"], hc["speed"], lam))
                errs.append(np.linalg.norm(pred - hc["C"], axis=1).mean())
            table[f"lam{lam}_s{sm}"] = float(np.mean(errs))
    best = min(table, key=table.get)
    lam, sm = best[3:].split("_s")
    return float(lam), float(sm), table


# ---------------------------------------------------------------- geometry ----------------------------------------

def geometry(d, dbin, sbin, theta, speed, k=64, n_boot=200, seed=0):
    knot, held = d["role"] == "knot", d["role"] != "knot"
    pca = mf.fit_pca(d["X"][knot], k)
    Z = pca.project(d["X"])
    kc = cell_table(Z, theta, speed, dbin, sbin, knot)
    hc = cell_table(Z, theta, speed, dbin, sbin, held)
    fit = tm.cone_cylinder_fit(kc["C"], kc["s"], kc["d"], N_DIR)
    lam, sm, _ = choose_sheet(Z, theta, speed, dbin, sbin, d["is_train"], (-1, ()))
    sheet = fit_sheet(kc, lam, sm)
    tot = ((hc["C"] - hc["C"].mean(0)) ** 2).sum()
    noise = (hc["se"] ** 2).sum()
    preds = {"cylinder": tm.predict_sheet(fit, hc["s"], hc["d"], "cylinder"),
             "cone": tm.predict_sheet(fit, hc["s"], hc["d"], "cone"),
             "sheet_tps": sheet(sheet_inputs(hc["theta"], hc["speed"], lam)),
             "speed_only": fit["m"][np.searchsorted(fit["bands"], hc["s"])]}
    r2 = {m: float(1 - ((p - hc["C"]) ** 2).sum() / tot) for m, p in preds.items()}
    r2["noise_ceiling"] = float(1 - noise / tot)
    # ring radius per speed bin, on train clips, noise-corrected; line fit r = a + b v with a clip bootstrap
    tr = d["is_train"]
    def radii(rows, raw=False):
        out = []
        for s in range(N_SPD):
            c = cell_table(Z, theta, speed, dbin, sbin, rows & (sbin == s))
            out.append(tm.ring_radius_corrected(c["C"], float(np.sqrt((c["se"] ** 2).mean())))[0 if raw else 1])
        return np.array(out)
    vb = np.array([speed[tr & (sbin == s)].mean() for s in range(N_SPD)])
    r = radii(tr)
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(tr)
    boot = []
    for _ in range(n_boot):
        pick = np.zeros(len(tr), int)
        np.add.at(pick, rng.choice(idx, len(idx)), 1)
        rows = pick > 0
        rb = radii(rows)                                      # unweighted resample (set of drawn clips)
        boot.append(np.r_[np.polyfit(vb, rb, 1)[::-1], rb[-1] / rb[0] if rb[0] > 0 else np.nan])
    boot = np.array(boot)
    a, b = np.polyfit(vb, r, 1)[::-1]
    sat = np.polyfit(vb[vb >= 1.4], r[vb >= 1.4], 1)[0] if (vb >= 1.4).sum() >= 2 else np.nan
    return {"heldout_cell_r2": r2, "sheet_lambda": lam, "sheet_smoothing": sm,
            "cone_r_by_band_rel": fit["r_cone"].tolist(),
            "radius_by_speed_bin": {"speed_mean": vb.tolist(), "radius_noise_corrected": r.tolist(),
                                    "radius_raw_rms": radii(tr, raw=True).tolist()},
            "radius_line": {"intercept_a": float(a), "slope_b": float(b),
                            "intercept_ci95": np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
                            "slope_ci95": np.percentile(boot[:, 1], [2.5, 97.5]).tolist(),
                            "a_over_r_at_mean_speed": float(a / r.mean()),
                            "slope_above_1p4_mps": float(sat)},
            "fast_over_slow_radius": {"mean": float(r[-1] / r[0]), "ci95": np.nanpercentile(boot[:, 2], [2.5, 97.5]).tolist(),
                                      "n_boot_slow_radius_zero": int(np.isnan(boot[:, 2]).sum()),
                                      "speed_ratio": float(vb[-1] / vb[0])}}


# ---------------------------------------------------------------- steering ----------------------------------------

def steer_seed(d, dbin, sbin, theta, speed, seed, k=64, K=21):
    block = BLOCKS[seed]
    held_d, held_s = [block[0], (block[0] + 1) % N_DIR], list(block[1])
    held = np.isin(dbin, held_d) & np.isin(sbin, held_s)
    knot, probe, test = (d["role"] == r for r in ("knot", "probe", "test"))
    kf = knot & ~held
    pca = mf.fit_pca(d["X"][kf], k)
    Z = pca.project(d["X"])
    lam, sm, sel = choose_sheet(Z, theta, speed, dbin, sbin, d["is_train"], block)
    kc = cell_table(Z, theta, speed, dbin, sbin, kf)
    sheet = fit_sheet(kc, lam, sm)
    pr = probe & ~held                                   # the probes never see the held-out cells either
    dprobe = mf.ProbeReadout(d["X"][pr], theta[pr], True)
    sprobe = mf.ProbeReadout(d["X"][pr], speed[pr], False)
    s_lin = np.linspace(0, 1, K)
    rows = []
    ids = d["df"]["id"].to_numpy()

    def score(task, arm, Zk, pick, tgt_theta, tgt_speed, real_mean):
        x = d["X"][pick].astype(float)
        W = pca.lift(Zk) + pca.complement(x)[:, None]
        n = len(pick)
        end = W[:, -1]
        th_hat, v_hat = dprobe.predict(end), sprobe.predict(end)
        th0, v0 = dprobe.predict(x), sprobe.predict(x)
        rad = np.linalg.norm(dprobe.raw(W.reshape(-1, W.shape[-1])).reshape(n, K, 2), axis=-1).min(1)
        rec = {"task": task, "arm": arm, "clip": ids[pick], "seed": np.full(n, seed),
               "nearest_real_R": mf.nearest_real_agreement(end, x, real_mean),
               "min_dir_radius": rad, "edit_norm": np.linalg.norm(end - x, axis=1)}
        if task == "direction":
            rec["endpoint_err"] = mf.value_error(th_hat, tgt_theta, True)
            rec["offtarget_change"] = np.abs(v_hat - v0)
        else:
            rec["endpoint_err"] = np.abs(v_hat - tgt_speed)
            rec["offtarget_change"] = mf.value_error(th_hat, th0, True)
        rows.append(rec)

    # (iii) joint: both variables to a held-out cell; global 1-D curves (ring pooled over speeds, speed line pooled
    # over directions), all fit on knot clips outside the held block
    labels_all = np.array([22.5 * b + 8.4375 for b in dbin[kf]])
    cent_ring = mf.centroids(Z[kf], labels_all)
    ring_all = mf.fit_curve(cent_ring, True, angle="labels", spline="smooth")
    cent_v = mf.centroids(Z[kf], speed[kf])
    line_all = mf.fit_curve(cent_v, False, spline="smooth", extend="linear")
    K1 = (K + 1) // 2

    def joint(pick, tgt_theta, tgt_speed, real_mean):
        x = d["X"][pick].astype(float)
        n = len(pick)
        th_s, v_s = theta[pick], speed[pick]
        Z0 = Z[pick]
        dth = mf.wrap_pi(np.radians(tgt_theta - th_s))
        path_th = np.degrees(np.radians(th_s)[:, None] + s_lin[None] * dth[:, None])
        path_v = v_s[:, None] + s_lin[None] * (tgt_speed - v_s)[:, None]
        P = sheet(sheet_inputs(path_th.ravel(), path_v.ravel(), lam)).reshape(n, K, -1)
        Zs = Z0[:, None] + P - P[:, :1]
        Zc = mf.linear_coords(Z0, P[:, 0], P[:, -1], K)
        s1 = np.linspace(0, 1, K1)
        ta, tb = ring_all.coord_of_value(th_s), ring_all.coord_of_value(np.full(n, tgt_theta))
        dR = mf.manifold_coords(np.zeros_like(Z0), ring_all, ta, tb, K1)          # [n, K1, k] displacement
        dV = mf.manifold_coords(np.zeros_like(Z0), line_all, v_s, np.full(n, tgt_speed), K1)
        Zdv = Z0[:, None] + np.concatenate([dR, dR[:, -1:] + dV[:, 1:]], 1)        # direction then speed
        Zvd = Z0[:, None] + np.concatenate([dV, dV[:, -1:] + dR[:, 1:]], 1)        # speed then direction
        # each arm's own parameter schedule (what the probes should read at waypoint k)
        sched_lin = (path_th, path_v)
        th_dv = np.concatenate([np.degrees(np.radians(th_s)[:, None] + s1[None] * dth[:, None]),
                                np.repeat(np.full((n, 1), tgt_theta), K - K1, 1)], 1)
        v_dv = np.concatenate([np.repeat(v_s[:, None], K1, 1),
                               v_s[:, None] + s1[None, 1:] * (tgt_speed - v_s)[:, None]], 1)
        th_vd = np.concatenate([np.repeat(th_s[:, None], K1, 1),
                                np.degrees(np.radians(th_s)[:, None] + s1[None, 1:] * dth[:, None])], 1)
        v_vd = np.concatenate([v_s[:, None] + s1[None] * (tgt_speed - v_s)[:, None],
                               np.repeat(np.full((n, 1), tgt_speed), K - K1, 1)], 1)
        th0, v0 = dprobe.predict(x), sprobe.predict(x)
        norm_sheet = None
        for arm, Zk, (sth, sv) in (("sheet", Zs, sched_lin), ("chord_sheet", Zc, sched_lin),
                                   ("seq_dir_then_speed", Zdv, (th_dv, v_dv)), ("seq_speed_then_dir", Zvd, (th_vd, v_vd))):
            W = pca.lift(Zk) + pca.complement(x)[:, None]
            Wf = W.reshape(-1, W.shape[-1])
            th_hat = dprobe.predict(Wf).reshape(n, K)
            v_hat = sprobe.predict(Wf).reshape(n, K)
            rad = np.linalg.norm(dprobe.raw(Wf).reshape(n, K, 2), axis=-1).min(1)
            en = np.linalg.norm(W[:, -1] - x, axis=1)
            if arm == "sheet":
                norm_sheet = en
            rec = {"task": "joint", "arm": arm, "clip": ids[pick], "seed": np.full(n, seed),
                   "endpoint_err_dir": mf.value_error(th_hat[:, -1], tgt_theta, True),
                   "endpoint_err_speed": np.abs(v_hat[:, -1] - tgt_speed),
                   "path_err_dir_vs_schedule": mf.value_error(th_hat, sth, True).mean(1),
                   "path_err_speed_vs_schedule": np.abs(v_hat - sv).mean(1),
                   "min_dir_radius": rad, "edit_norm": en, "edit_norm_ratio_to_sheet": en / norm_sheet,
                   "nearest_real_R": mf.nearest_real_agreement(W[:, -1], x, real_mean)}
            if arm == "seq_dir_then_speed":
                rec["offtarget_speed_during_dir_phase"] = np.abs(v_hat[:, :K1] - v0[:, None]).mean(1)
                rec["offtarget_dir_during_speed_phase"] = mf.value_error(th_hat[:, K1 - 1:], th_hat[:, K1 - 1:K1], True).mean(1)
            if arm == "seq_speed_then_dir":
                rec["offtarget_dir_during_speed_phase"] = mf.value_error(th_hat[:, :K1], th0[:, None], True).mean(1)
                rec["offtarget_speed_during_dir_phase"] = np.abs(v_hat[:, K1 - 1:] - v_hat[:, K1 - 1:K1]).mean(1)
            rows.append(rec)

    for dd in held_d:
        for ss in held_s:
            cell = test & (dbin == dd) & (sbin == ss)
            if cell.sum() == 0:
                continue
            real_mean = d["X"][cell].mean(0)
            tgt_theta = float(np.degrees(np.angle(np.exp(1j * np.radians(theta[cell])).mean())) % 360)
            tgt_speed = float(speed[cell].mean())
            pick = np.flatnonzero(test & ~np.isin(dbin, held_d) & ~np.isin(sbin, held_s))
            if len(pick):
                joint(pick, tgt_theta, tgt_speed, real_mean)
            # (i) direction at fixed speed
            pick = np.flatnonzero(test & (sbin == ss) & ~np.isin(dbin, held_d))
            if len(pick):
                th_s, v_s = theta[pick], speed[pick]
                dth = mf.wrap_pi(np.radians(tgt_theta - th_s))
                path_th = np.degrees(np.radians(th_s)[:, None] + s_lin[None] * dth[:, None])
                P = sheet(sheet_inputs(path_th.ravel(), np.repeat(v_s, K), lam)).reshape(len(pick), K, -1)
                Z0 = Z[pick]
                Zs = Z0[:, None] + P - P[:, :1]
                band = kf & (sbin == ss)
                labels = np.array([22.5 * b + 8.4375 for b in dbin[band]])
                cent = mf.centroids(Z[band], labels)
                ring = mf.fit_curve(cent, True, angle="labels", spline="smooth")
                Zr = mf.manifold_coords(Z0, ring, ring.coord_of_value(th_s), ring.coord_of_value(tgt_theta), K)
                raw = mf.raw_knot_curve(ring, cent)
                Zc = mf.linear_coords(Z0, P[:, 0], P[:, -1], K)
                Zraw = Z0[:, None] + s_lin[None, :, None] * (mf.piecewise_linear_point(raw, np.full(len(pick), tgt_theta))
                                                             - mf.piecewise_linear_point(raw, th_s))[:, None]
                for arm, Zk in (("sheet", Zs), ("ring1d", Zr), ("chord_sheet", Zc), ("chord_raw", Zraw), ("unedited", np.repeat(Z0[:, None], K, 1))):
                    score("direction", arm, Zk, pick, tgt_theta, tgt_speed, real_mean)
            # (ii) speed at fixed direction
            pick = np.flatnonzero(test & (dbin == dd) & ~np.isin(sbin, held_s))
            if len(pick):
                th_s, v_s = theta[pick], speed[pick]
                path_v = v_s[:, None] + s_lin[None] * (tgt_speed - v_s)[:, None]
                P = sheet(sheet_inputs(np.repeat(th_s, K), path_v.ravel(), lam)).reshape(len(pick), K, -1)
                Z0 = Z[pick]
                Zs = Z0[:, None] + P - P[:, :1]
                cent = mf.centroids(Z[kf], speed[kf])
                line = mf.fit_curve(cent, False, spline="smooth", extend="linear")
                Zl = mf.manifold_coords(Z0, line, v_s, np.full(len(pick), tgt_speed), K)
                raw = mf.raw_knot_curve(line, cent)
                Zc = mf.linear_coords(Z0, P[:, 0], P[:, -1], K)
                Zraw = Z0[:, None] + s_lin[None, :, None] * (mf.piecewise_linear_point(raw, np.full(len(pick), tgt_speed))
                                                             - mf.piecewise_linear_point(raw, v_s))[:, None]
                for arm, Zk in (("sheet", Zs), ("speedline", Zl), ("chord_sheet", Zc), ("chord_raw", Zraw), ("unedited", np.repeat(Z0[:, None], K, 1))):
                    score("speed", arm, Zk, pick, tgt_theta, tgt_speed, real_mean)
    return {"block": {"held_dir_bins": held_d, "held_speed_bins": held_s}, "lambda": lam, "smoothing": sm,
            "selection_table": sel, "rows": rows}


def aggregate(seed_outs, n_boot=1000):
    metrics = ("endpoint_err", "offtarget_change", "nearest_real_R", "min_dir_radius", "edit_norm")
    out = {}
    for task, arms in (("direction", ("sheet", "ring1d", "chord_sheet", "chord_raw", "unedited")),
                       ("speed", ("sheet", "speedline", "chord_sheet", "chord_raw", "unedited"))):
        cat = {}
        for a in arms:
            recs = [r for so in seed_outs for r in so["rows"] if r["task"] == task and r["arm"] == a]
            cat[a] = {q: np.concatenate([r[q] for r in recs]) for q in (*metrics, "clip", "seed")}
        res = {"n_steers": int(len(cat["sheet"]["clip"])), "n_clips": int(len(np.unique(cat["sheet"]["clip"]))),
               "summary": {a: {q: float(cat[a][q].mean()) for q in metrics} for a in arms}, "gaps": {}}
        for other in arms[1:]:
            res["gaps"][f"sheet_minus_{other}"] = {
                q: tm.cluster_bootstrap(cat["sheet"][q] - cat[other][q], cat["sheet"]["clip"], n_boot=n_boot)
                for q in metrics if not (other == "unedited" and q == "edit_norm")}
        res["per_seed_endpoint_gap_vs_1d"] = [
            float((cat["sheet"]["endpoint_err"] - cat[arms[1]]["endpoint_err"])[cat["sheet"]["seed"] == s].mean())
            for s in sorted(set(cat["sheet"]["seed"].tolist()))]
        out[task] = res
    jm = ("endpoint_err_dir", "endpoint_err_speed", "path_err_dir_vs_schedule", "path_err_speed_vs_schedule",
          "min_dir_radius", "edit_norm", "edit_norm_ratio_to_sheet", "nearest_real_R")
    arms = ("sheet", "chord_sheet", "seq_dir_then_speed", "seq_speed_then_dir")
    cat = {}
    for a in arms:
        recs = [r for so in seed_outs for r in so["rows"] if r["task"] == "joint" and r["arm"] == a]
        cat[a] = {q: np.concatenate([r[q] for r in recs]) for q in recs[0] if q not in ("task", "arm")}
    res = {"n_steers": int(len(cat["sheet"]["clip"])), "n_clips": int(len(np.unique(cat["sheet"]["clip"]))),
           "summary": {a: {q: float(v.mean()) for q, v in cat[a].items() if q not in ("clip", "seed")} for a in arms},
           "gaps": {}}
    for other in arms[1:]:
        res["gaps"][f"sheet_minus_{other}"] = {q: tm.cluster_bootstrap(cat["sheet"][q] - cat[other][q], cat["sheet"]["clip"],
                                                                       n_boot=n_boot) for q in jm if q != "edit_norm_ratio_to_sheet"}
        res["gaps"][f"sheet_minus_{other}"]["edit_norm_ratio_other_over_sheet"] = tm.cluster_bootstrap(
            cat[other]["edit_norm_ratio_to_sheet"], cat["sheet"]["clip"], n_boot=n_boot)
    for a in arms[2:]:
        res["summary"][a + "_offtarget_ci"] = {q: tm.cluster_bootstrap(cat[a][q], cat[a]["clip"], n_boot=n_boot)
                                               for q in ("offtarget_speed_during_dir_phase", "offtarget_dir_during_speed_phase")}
    out["joint"] = res
    return out


def run_layer(layer, seeds, k, act_dir=None):
    d = load_inputs("speed", layer, act_dir=act_dir)
    theta, speed = d["df"]["theta_degrees"].to_numpy(float), d["df"]["speed_mps"].to_numpy(float)
    dbin, sbin = bins(theta, speed)
    geo = geometry(d, dbin, sbin, theta, speed, k)
    so = [steer_seed(d, dbin, sbin, theta, speed, s, k) for s in seeds]
    return layer, {"geometry": geo, "steer": aggregate(so),
                   "seeds": [{kk: v for kk, v in s.items() if kk != "rows"} for s in so]}


def figure(res, path):
    L = sorted(res["layers"], key=int)
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    a = ax[0]
    for l in L:
        g = res["layers"][l]["geometry"]["radius_by_speed_bin"]
        a.plot(g["speed_mean"], g["radius_noise_corrected"], "o-", label=f"point {l}")
    a.set_xlabel("speed (m/s)"); a.set_ylabel("ring radius (PCA-64 units, noise-corrected)")
    a.set_title("Ring radius vs speed: cone would pass through 0", fontsize=9); a.set_xlim(0, None); a.set_ylim(0, None)
    a.legend(fontsize=8)
    for i, (task, other, unit) in enumerate((("direction", "ring1d", "deg"), ("speed", "speedline", "m/s"))):
        a = ax[i + 1]
        arms = ("unedited", "chord_raw", "chord_sheet", other, "sheet")
        w = 0.16
        for j, arm in enumerate(arms):
            a.bar(np.arange(len(L)) + (j - 2) * w, [res["layers"][l]["steer"][task]["summary"][arm]["endpoint_err"] for l in L],
                  w, label=arm)
        a.set_xticks(range(len(L))); a.set_xticklabels([f"pt {l}" for l in L])
        a.set_ylabel(f"endpoint probe error ({unit})")
        a.set_title(f"({'i' if i == 0 else 'ii'}) steer {task} to a held-out (θ, v) cell", fontsize=9)
        a.legend(fontsize=7)
    a = ax[3]
    arms = ("sheet", "chord_sheet", "seq_dir_then_speed", "seq_speed_then_dir")
    w = 0.2
    for j, arm in enumerate(arms):
        sm = [res["layers"][l]["steer"]["joint"]["summary"][arm] for l in L]
        a.bar(np.arange(len(L)) + (j - 1.5) * w, [m["path_err_dir_vs_schedule"] for m in sm], w, label=arm)
    a.set_xticks(range(len(L))); a.set_xticklabels([f"pt {l}" for l in L])
    a.set_ylabel("direction-probe error vs the arm's schedule, path mean (deg)")
    a.set_title("(iii) joint steer to a held-out (θ, v) cell: sheet vs chord vs sequential 1-D", fontsize=9)
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[12, 19, 22])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--random-init", action="store_true", help="also run the random-init encoder's meanpool (key random_init)")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "p5_velocity_sheet.json"))
    ap.add_argument("--fig", default=str(PROJECT_ROOT / "figures" / "fig_velocity_sheet.png"))
    args = ap.parse_args()
    t0 = time.time()
    rand_dir = PROJECT_ROOT / "artifacts" / "activations" / "speed" / "random"
    cfg = [(l, None) for l in args.layers] + ([(l, rand_dir) for l in args.layers] if args.random_init else [])
    outs = Parallel(n_jobs=args.jobs)(delayed(run_layer)(l, args.seeds, args.k, a) for l, a in cfg)
    res = {"layers": {str(l): o for (l, o), (_, a) in zip(outs, cfg) if a is None},
           "random_init": {str(l): o for (l, o), (_, a) in zip(outs, cfg) if a is not None}}
    res["keys"] = {
        "layers[p].geometry.heldout_cell_r2": "fraction of the variance of held-out (non-knot) cell centroids (16 dir x 8 speed bins, PCA-64 from knot clips) explained by models fit on knot cell centroids: cylinder m(v)+u(theta), cone m(v)+r(v)u(theta), sheet_tps, speed_only; noise_ceiling = 1 - (sum of held-out centroid squared SEs)/total",
        "layers[p].geometry.radius_by_speed_bin": "ring radius (RMS distance of the 16 direction-bin centroids from their mean) per speed bin on train clips, centroid noise subtracted in quadrature; radius_line = least-squares r = a + b v (clip-bootstrap CIs); a cone (radius proportional to speed) needs a ~ 0 and fast_over_slow_radius ~ speed_ratio; a cylinder needs b ~ 0",
        "layers[p].steer.{direction,speed}": "held-out blocks per seed (2 dir bins x 2 interior speed bins; seeds 0-3 pooled). endpoint_err: direction task = direction-probe error to the target cell's mean theta (deg); speed task = speed-probe error to the target cell's mean speed (m/s). offtarget_change: |change| of the other probe vs the unedited clip (m/s for direction steers, deg for speed steers). nearest_real_R = 1 - |x'-mu|/|x-mu| with mu the mean of test clips in the target cell. min_dir_radius: min norm of the direction probe's (sin, cos) output along the K=21 waypoints. gaps sheet_minus_X: mean paired difference with 95% clip-bootstrap CI.",
        "random_init": "same analysis on the random-init encoder's meanpool (artifacts/activations/speed/random), same split and seeds",
        "layers[p].steer.joint": "(iii) both variables to a held-out cell (carriers: test clips outside the held direction AND speed bins). Arms: sheet (TPS walked along the straight line in (theta, v)), chord_sheet (straight line in PCA-64 between the sheet's endpoints; endpoint = sheet's), seq_dir_then_speed / seq_speed_then_dir (additive 1-D edits: global ring pooled over speeds, then speed line pooled over directions, or the reverse; the two orders share the endpoint because the edits add). endpoint_err_dir (deg) / endpoint_err_speed (m/s) vs the target cell's means; path_err_*_vs_schedule = mean over the K=21 waypoints of |probe - the arm's own scheduled value|; offtarget_speed_during_dir_phase / offtarget_dir_during_speed_phase (sequential arms): mean |change| of the unmoved variable during each phase; edit_norm_ratio_other_over_sheet = |edit_other| / |edit_sheet| per clip. Probes are fit on probe clips outside the held cells (all tasks).",
        "arms": "sheet = TPS over (cos, sin, lambda*(v-2.125)/1.875) walked along the one variable; ring1d = periodic smoothing spline over the target speed bin's knot direction-bin centroids; speedline = natural smoothing spline over knot speed-value centroids pooled over directions; chord_sheet = straight line between the sheet's endpoints (endpoint-matched: endpoint metrics equal the sheet's); chord_raw = straight line between the raw-centroid polyline points of the 1-D curve (A.9 chord); unedited = no edit",
    }
    res["provenance"] = provenance(seeds={"blocks": args.seeds, "bootstrap": 0}, layers=args.layers, k=args.k,
                                   pool="meanpool", blocks=BLOCKS, n_dir_bins=N_DIR, n_speed_bins=N_SPD,
                                   lambda_grid=LAMBDAS, smoothing_grid=SMOOTHINGS, wall_s=round(time.time() - t0, 1))
    Path(args.out).write_text(json.dumps(res, indent=1))
    figure(res, args.fig)
    print("wrote", args.out, args.fig, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

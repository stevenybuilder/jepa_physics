"""Velocity sheet v2: hold-out audit, block / cross hold-outs, norm-matched baselines, cone decomposition, dose and
extrapolation, and the acceleration x direction sheet. Encoder level (pooled activations), CPU only.

Builds on scripts/run_velocity_sheet.py (same split roles, bins, TPS sheet, selection of lambda/smoothing):
  knot clips (folds 0-2) outside the held cells -> PCA-64, cell centroids, TPS sheet, every 1-D curve, the probe-edit
  ridge; probe clips (folds 3-4) outside the held cells -> every READER; test clips -> carriers and the nearest-real
  target mean (test clips inside the target cell, never carriers).
Designs (per seed 0-3): block2 (v1: 2 dir bins x 2 speed bins), block3 (3 x 3), cross (2 dir bins at all speeds AND
2 speed bins at all directions unseen: target direction and speed both unseen anywhere in fitting).
Tasks: joint (both variables to a target cell; carriers = test clips outside every held dir and speed bin),
direction-only (block designs; carriers in the target speed bin), speed-only (block designs; carriers in the target
direction bin). Conditions: own norm and matched (each arm's endpoint displacement rescaled to the sheet's per clip).
Readers: ridge direction (sin, cos) and speed on probe folds 3+4, the same on fold 3 only and fold 4 only (disjoint
reader check), an MLP reader on folds 3+4, ridge start-position (x, y) for the off-target position leak, nearest-real R.

  python scripts/run_velocity_sheet_v2.py --dataset speed --layers 12 22 --jobs 4
  python scripts/run_velocity_sheet_v2.py --dataset acceleration --layers 12 21 --jobs 4 --out results/p5_accel_sheet_v2_part.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as skPCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_velocity_sheet import N_DIR, N_SPD, LAMBDAS, SMOOTHINGS, cell_table, fit_sheet  # noqa: E402
from wm import manifold as mf  # noqa: E402
from wm import timeman as tm  # noqa: E402
from wm import vsheet as vs  # noqa: E402
from wm.data import PROJECT_ROOT  # noqa: E402
from wm.p2_data import load_inputs  # noqa: E402
from wm.provenance import provenance  # noqa: E402

COL = {"speed": "speed_mps", "acceleration": "acceleration_mps2"}
DOSES = (0.5, 0.75, 1.0, 1.5, 2.0)


def bins(theta, v):
    sv = np.sort(np.unique(v))
    return tm.angular_bins(theta, N_DIR), np.searchsorted(sv, v) * N_SPD // len(sv)


class Sheet:
    def __init__(self, cells, lam, sm, vmid, vhalf):
        self.lam, self.vmid, self.vhalf = lam, vmid, vhalf
        self.f = fit_sheet_inputs(cells, lam, sm, vmid, vhalf)

    def inp(self, th, v):
        th = np.radians(np.asarray(th, float))
        return np.stack([np.cos(th), np.sin(th), self.lam * (np.asarray(v, float) - self.vmid) / self.vhalf], -1)

    def __call__(self, th, v):
        return self.f(self.inp(th, v))


def fit_sheet_inputs(cells, lam, sm, vmid, vhalf):
    from scipy.interpolate import RBFInterpolator
    th = np.radians(cells["theta"])
    x = np.stack([np.cos(th), np.sin(th), lam * (cells["speed"] - vmid) / vhalf], -1)
    return RBFInterpolator(x, cells["C"], kernel="thin_plate_spline", smoothing=sm)


def choose(Z, theta, v, dbin, sbin, rows, design, seed, vmid, vhalf):
    """lambda, smoothing by reconstruction of the OTHER seeds' held cells of the same design, with this seed's held
    clips removed first (so the evaluation block never enters selection)."""
    held_eval = vs.holdout(dbin, sbin, design, seed)[0]
    rows = rows & ~held_eval
    table = {}
    for lam in LAMBDAS:
        for sm in SMOOTHINGS:
            errs = []
            for s2 in range(4):
                if s2 == seed:
                    continue
                h2, tg2, _, _ = vs.holdout(dbin, sbin, design, s2)
                tcell = np.zeros_like(h2)
                for (dd, ss) in tg2:
                    tcell |= (dbin == dd) & (sbin == ss)
                kc = cell_table(Z, theta, v, dbin, sbin, rows & ~h2)
                hc = cell_table(Z, theta, v, dbin, sbin, rows & tcell)
                if len(hc["C"]) == 0:
                    continue
                pred = fit_sheet_inputs(kc, lam, sm, vmid, vhalf)(
                    np.stack([np.cos(np.radians(hc["theta"])), np.sin(np.radians(hc["theta"])),
                              lam * (hc["speed"] - vmid) / vhalf], -1))
                errs.append(np.linalg.norm(pred - hc["C"], axis=1).mean())
            table[f"lam{lam}_s{sm}"] = float(np.mean(errs))
    best = min(table, key=table.get)
    lam, sm = best[3:].split("_s")
    return float(lam), float(sm), table


class Readers:
    def __init__(self, d, theta, v, rows, vmid, vhalf, seed):
        X = d["X"]
        f = d["fold"]
        self.vmid, self.vhalf = vmid, vhalf
        self.dir = mf.ProbeReadout(X[rows], theta[rows], True)
        self.spd = mf.ProbeReadout(X[rows], v[rows], False)
        self.dir3 = mf.ProbeReadout(X[rows & (f == 3)], theta[rows & (f == 3)], True)
        self.spd3 = mf.ProbeReadout(X[rows & (f == 3)], v[rows & (f == 3)], False)
        self.dir4 = mf.ProbeReadout(X[rows & (f == 4)], theta[rows & (f == 4)], True)
        self.spd4 = mf.ProbeReadout(X[rows & (f == 4)], v[rows & (f == 4)], False)
        pos = d["df"][["start_x", "start_y"]].to_numpy(float)
        self.pos = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 15))).fit(X[rows], pos[rows])
        th = np.radians(theta[rows])
        Y = np.stack([np.sin(th), np.cos(th), (v[rows] - vmid) / vhalf], 1)
        self.mlp = make_pipeline(StandardScaler(), skPCA(64, whiten=True, random_state=seed),
                                 MLPRegressor(hidden_layer_sizes=(128,), alpha=1.0, solver="lbfgs", max_iter=3000,
                                              random_state=seed)).fit(X[rows], Y)

    def read(self, X):
        o = self.mlp.predict(X)
        return {"dir": self.dir.predict(X), "spd": self.spd.predict(X), "dir3": self.dir3.predict(X),
                "spd3": self.spd3.predict(X), "dir4": self.dir4.predict(X), "spd4": self.spd4.predict(X),
                "pos": self.pos.predict(X), "dir_mlp": np.degrees(np.arctan2(o[:, 0], o[:, 1])) % 360,
                "spd_mlp": o[:, 2] * self.vhalf + self.vmid,
                "dir_rad": np.linalg.norm(self.dir.raw(X), axis=-1)}


def curve_delta(curve, a, b):
    """Endpoint displacement along a 1-D curve from coordinate a to b (per row)."""
    return mf.manifold_coords(np.zeros((len(a), curve(np.array([a[0]])).shape[-1])), curve, a, b, 2)[:, -1]


def ring_from(Z, rows, dbin, spline="smooth"):
    """1-D direction ring over direction-bin centroids, labels angle (value-ordered knots). spline: "smooth" (our
    stored FITPACK splrep smoother, s = m; known to cost 1-7 deg at held-out endpoints, REPORT 4.3) or "interp"
    (the paper's interpolating periodic cubic through the raw centroids, Goodfire A.3)."""
    lab = 22.5 * dbin[rows] + 8.4375            # mean theta of the bin (as v1)
    return mf.fit_curve(mf.centroids(Z[rows], lab), True, angle="labels", spline=spline)


def line_from(Z, rows, sbin, v, spline="smooth"):
    lab = np.array([v[rows & (sbin == s)].mean() for s in sbin[rows]])
    return mf.fit_curve(mf.centroids(Z[rows], lab), False, spline=spline, extend="linear")


_DIAG = None


def reinsch_ring(Z, rows, dbin):
    """causalab Reinsch smoothing ring (the authors' SplineManifold) with lambda chosen by leave-block-out CV on
    the kept knots (scripts/run_endpoint_diagnosis.py cv_lambda / causalab_curve, reused). Returns f(theta_deg)."""
    global _DIAG
    if _DIAG is None:
        import importlib.util
        import torch
        torch.set_num_threads(1)
        spec = importlib.util.spec_from_file_location("run_endpoint_diagnosis",
                                                      Path(__file__).with_name("run_endpoint_diagnosis.py"))
        _DIAG = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_DIAG)
    lab = 22.5 * dbin[rows] + 8.4375
    cent = mf.centroids(Z[rows], lab)
    coords = np.radians(cent["values"])
    cv = _DIAG.cv_lambda(coords, cent["C"], cent["values"], block=2, margin=1)
    f, _ = _DIAG.causalab_curve(coords, cent["C"], cv["lambda_cv"])
    return (lambda th: f(np.radians(np.asarray(th, float)) % (2 * np.pi))), cv["lambda_cv"]


def ring_delta(ring, th_a, th_b):
    return curve_delta(ring, ring.coord_of_value(th_a), ring.coord_of_value(th_b))


def line_delta(line, va, vb):
    return curve_delta(line, np.asarray(va, float), np.asarray(vb, float))


def nearest_seen_bins(target, seen, periodic_n=None):
    seen = np.asarray(sorted(set(seen)))
    dist = np.abs(seen - target)
    if periodic_n:
        dist = np.minimum(dist % periodic_n, periodic_n - dist % periodic_n)
    return seen[dist == dist.min()].tolist()


def run_seed(dataset, layer, design, seed, k=64, dose=False):
    d = load_inputs(dataset, layer)
    theta = d["df"]["theta_degrees"].to_numpy(float)
    v = d["df"][COL[dataset]].to_numpy(float)
    vmid, vhalf = (v.min() + v.max()) / 2, (v.max() - v.min()) / 2
    dbin, sbin = bins(theta, v)
    held, targets, hd, hs = vs.holdout(dbin, sbin, design, seed)
    knot, probe, test = (d["role"] == r for r in ("knot", "probe", "test"))
    kf = knot & ~held
    pca = mf.fit_pca(d["X"][kf], k)
    Z = pca.project(d["X"])
    lam, sm, sel = choose(Z, theta, v, dbin, sbin, d["is_train"], design, seed, vmid, vhalf)
    kc = cell_table(Z, theta, v, dbin, sbin, kf)
    P = Sheet(kc, lam, sm, vmid, vhalf)
    P0 = Sheet(kc, lam, 0.0, vmid, vhalf)          # interpolating thin-plate variant (smoothing 0, same lambda)
    frame = vs.ConeFrame(P)
    vgrid = np.linspace(v.min(), v.max(), 16)

    def pooled_ring(th):
        """The sheet averaged over speed: P_bar(theta) = mean_v P(theta, v) (a cylinder ring with the sheet's own
        smoothing), so sheet - pooled isolates the speed-specific slice from the TPS pooling/denoising."""
        th = np.asarray(th, float)
        return P(np.repeat(th, len(vgrid)), np.tile(vgrid, len(th))).reshape(len(th), len(vgrid), -1).mean(1)
    R = Readers(d, theta, v, probe & ~held, vmid, vhalf, seed)
    # probe-edit ridge in PCA-64 on knot clips (edit construction data, disjoint from the readers)
    th = np.radians(theta[kf])
    pe = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Z[kf], np.stack([np.sin(th), np.cos(th), (v[kf] - vmid) / vhalf], 1))
    ring_all = ring_from(Z, kf, dbin)
    line_all = line_from(Z, kf, sbin, v)
    ring_all_i = ring_from(Z, kf, dbin, "interp")
    line_all_i = line_from(Z, kf, sbin, v, "interp")
    ring_all_r, lam_r_all = reinsch_ring(Z, kf, dbin)
    seen_s = np.unique(sbin[kf])
    ids = d["df"]["id"].to_numpy()
    X = d["X"].astype(float)
    cell_C = {(a, b): Z[kf & (dbin == a) & (sbin == b)].mean(0) for a in range(N_DIR) for b in range(N_SPD)
              if (kf & (dbin == a) & (sbin == b)).sum() >= 2}
    cd, cs = np.array([c[0] for c in cell_C]), np.array([c[1] for c in cell_C])
    CC = np.array(list(cell_C.values()))
    rows_out = []

    def ring_band(s_list):
        return ring_from(Z, kf & np.isin(sbin, s_list), dbin)

    def line_dirwin(dcenter):
        win = [(dcenter + j) % N_DIR for j in (-1, 0, 1)]
        return line_from(Z, kf & np.isin(dbin, win), sbin, v)

    def evaluate(task, pick, deltas, tgt_th, tgt_v, real_mean, cell, own_norm_ref="sheet", extra=None):
        x = X[pick]
        r0 = R.read(x)
        base = np.linalg.norm(deltas[own_norm_ref], axis=1)
        for cond in ("own", "matched"):
            for arm, dZ in deltas.items():
                if cond == "matched" and arm == "unedited":
                    continue
                dz = vs.rescale_rows(dZ, base) if cond == "matched" else dZ
                xe = x + pca.lift_delta(dz)
                r = R.read(xe)
                rec = {"task": task, "cond": cond, "arm": arm, "clip": ids[pick], "seed": np.full(len(pick), seed),
                       "cell": np.full(len(pick), cell[0] * 100 + cell[1]),
                       "edit_norm": np.linalg.norm(dz, axis=1),
                       "err_dir": mf.value_error(r["dir"], tgt_th, True), "err_spd": np.abs(r["spd"] - tgt_v),
                       "err_dir_f3": mf.value_error(r["dir3"], tgt_th, True), "err_spd_f3": np.abs(r["spd3"] - tgt_v),
                       "err_dir_f4": mf.value_error(r["dir4"], tgt_th, True), "err_spd_f4": np.abs(r["spd4"] - tgt_v),
                       "err_dir_mlp": mf.value_error(r["dir_mlp"], tgt_th, True),
                       "err_spd_mlp": np.abs(r["spd_mlp"] - tgt_v),
                       "leak_spd": np.abs(r["spd"] - r0["spd"]), "leak_dir": mf.value_error(r["dir"], r0["dir"], True),
                       "leak_pos": np.linalg.norm(r["pos"] - r0["pos"], axis=1),
                       "dir_radius": r["dir_rad"],
                       "nearest_real_R": mf.nearest_real_agreement(xe, x, real_mean)}
                rows_out.append(rec)
        if extra:
            extra(x, r0)

    for (dd, ss) in targets:
        cell = test & (dbin == dd) & (sbin == ss)
        if cell.sum() == 0:
            continue
        real_mean = X[cell].mean(0)
        tth, tv = vs.circ_mean_deg(theta[cell]), float(v[cell].mean())
        rreal = R.read(X[cell])
        rows_out.append({"task": "reader_floor", "cond": "real", "arm": "real_target_clips", "clip": ids[cell],
                         "seed": np.full(cell.sum(), seed), "cell": np.full(cell.sum(), dd * 100 + ss),
                         "err_dir": mf.value_error(rreal["dir"], tth, True), "err_spd": np.abs(rreal["spd"] - tv),
                         "err_dir_mlp": mf.value_error(rreal["dir_mlp"], tth, True),
                         "err_spd_mlp": np.abs(rreal["spd_mlp"] - tv)})
        tgt_seen_s = nearest_seen_bins(ss, seen_s)
        # ---------------- joint
        pick = np.flatnonzero(test & ~np.isin(dbin, hd) & ~np.isin(sbin, hs))
        n = len(pick)
        ths, v_s, Z0 = theta[pick], v[pick], Z[pick]
        tT, tV = np.full(n, tth), np.full(n, tv)
        dl = {}
        dl["sheet"] = P(tT, tV) - P(ths, v_s)
        dl["seq_global"] = ring_delta(ring_all, ths, tT) + line_delta(line_all, v_s, tV)
        dl["sheet_tps_interp"] = P0(tT, tV) - P0(ths, v_s)
        dl["seq_global_interp"] = ring_delta(ring_all_i, ths, tT) + line_delta(line_all_i, v_s, tV)
        dl["seq_global_reinschcv_ring_interp_line"] = (ring_all_r(tT) - ring_all_r(ths)) + line_delta(line_all_i, v_s, tV)
        dv, vd, rsrc = np.zeros((n, k)), np.zeros((n, k)), np.zeros((n, k))
        rtgt = ring_band(tgt_seen_s)
        ltgt = line_dirwin(dd)
        for sb in np.unique(sbin[pick]):
            m = sbin[pick] == sb
            rs = ring_band([sb])
            rsrc[m] = ring_delta(rs, ths[m], tT[m])
        for db in np.unique(dbin[pick]):
            m = dbin[pick] == db
            vd[m] = line_delta(line_dirwin(db), v_s[m], tV[m])
        dv = rsrc + line_delta(ltgt, v_s, tV)
        vd = vd + ring_delta(rtgt, ths, tT)
        dl["seq_local_dir_then_speed"] = dv
        dl["seq_local_speed_then_dir"] = vd
        dl["ring_src_only"] = rsrc
        src_c = np.array([cell_C.get((a, b), np.full(k, np.nan)) for a, b in zip(dbin[pick], sbin[pick])])
        okc = np.isfinite(src_c).all(1)
        src_c[~okc] = Z0[~okc]
        dl["chord_cells_idw"] = vs.idw_cell_estimate(cd, cs, CC, dd, ss) - src_c
        orc = knot & (dbin == dd) & (sbin == ss)
        dl["chord_cells_ORACLE"] = Z[orc].mean(0) - src_c
        tt = np.radians(tth)
        dl["probe_edit"] = vs.min_norm_probe_edit(pe.coef_, pe.intercept_, Z0,
                                                  np.tile([np.sin(tt), np.cos(tt), (tv - vmid) / vhalf], (n, 1)))
        dl["sheet_at_probe_edit_norm"] = vs.rescale_rows(dl["sheet"], np.linalg.norm(dl["probe_edit"], axis=1))
        dl["abl_radius_frozen"] = vs.factorial_delta(frame, ths, v_s, tT, tV, radius=False, shape=True)
        dl["abl_shape_frozen"] = vs.factorial_delta(frame, ths, v_s, tT, tV, radius=True, shape=False)
        dl["abl_global_speed_axis"] = vs.factorial_delta(frame, ths, v_s, tT, tV, radius=False, shape=False)
        dl["abl_cylinder_from_sheet"] = (frame.center(tV) - frame.center(v_s)) + pooled_ring(tT) - pooled_ring(ths)
        dl["unedited"] = np.zeros((n, k))

        def joint_path(x, r0):
            K = 11
            s_ = np.linspace(0, 1, K)
            dth = mf.wrap_pi(np.radians(tth - ths))
            pth = np.degrees(np.radians(ths)[:, None] + s_[None] * dth[:, None])
            pv = v_s[:, None] + s_[None] * (tv - v_s)[:, None]
            Pp = P(pth.ravel(), pv.ravel()).reshape(n, K, -1)
            Ds = Pp - Pp[:, :1]
            Dc = s_[None, :, None] * Ds[:, -1:]
            for arm, D in (("sheet", Ds), ("chord_sheet", Dc)):
                W = (x[:, None] + pca.lift_delta(D)).reshape(n * K, -1)
                rr = R.read(W)
                rows_out.append({"task": "joint_path", "cond": "own", "arm": arm, "clip": ids[pick],
                                 "seed": np.full(n, seed), "cell": np.full(n, dd * 100 + ss),
                                 "path_err_dir": mf.value_error(rr["dir"].reshape(n, K), pth, True).mean(1),
                                 "path_err_spd": np.abs(rr["spd"].reshape(n, K) - pv).mean(1),
                                 "min_dir_radius": rr["dir_rad"].reshape(n, K).min(1)})
        evaluate("joint", pick, dl, tth, tv, real_mean, (dd, ss), extra=joint_path)
        if design == "cross":
            continue
        # ---------------- direction only (carriers at the target speed bin)
        pick = np.flatnonzero(test & (sbin == ss) & ~np.isin(dbin, hd))
        if len(pick):
            n = len(pick)
            ths, v_s = theta[pick], v[pick]
            tT = np.full(n, tth)
            dl = {"sheet": P(tT, v_s) - P(ths, v_s)}
            rb = ring_band([ss])
            dl["ring1d_band"] = ring_delta(rb, ths, tT)
            dl["ring1d_band_interp"] = ring_delta(ring_from(Z, kf & (sbin == ss), dbin, "interp"), ths, tT)
            rr, _ = reinsch_ring(Z, kf & (sbin == ss), dbin)
            dl["ring1d_band_reinschcv"] = rr(tT) - rr(ths)
            dl["ring_global_interp"] = ring_delta(ring_all_i, ths, tT)
            dl["ring_global_reinschcv"] = ring_all_r(tT) - ring_all_r(ths)
            dl["sheet_tps_interp"] = P0(tT, v_s) - P0(ths, v_s)
            rg = ring_delta(ring_all, ths, tT)
            dl["ring_global"] = rg
            dl["ring_global_radius_scaled"] = rg * (frame.radius(np.array([tv]))[0] /
                                                    frame.radius(np.array([v[kf].mean()]))[0])
            pooled = [b for b in (ss - 1, ss, ss + 1) if b in seen_s]
            dl["ring_band_pm1"] = ring_delta(ring_band(pooled), ths, tT)
            dl["sheet_pooled_over_speed"] = pooled_ring(tT) - pooled_ring(ths)
            dl["sheet_slice_other_speed"] = P(tT, 2 * vmid - v_s) - P(ths, 2 * vmid - v_s)
            dl["unedited"] = np.zeros((n, k))

            def dose_dir(x, r0, dl=dl, n=n, ths=ths, pick=pick):
                if not dose:
                    return
                for arm in ("sheet", "ring1d_band", "ring_global", "sheet_pooled_over_speed"):
                    for a in DOSES:
                        r = R.read(x + pca.lift_delta(a * dl[arm]))
                        intended = mf.wrap_pi(np.radians(tth - ths))
                        got = mf.wrap_pi(np.radians(r["dir"] - r0["dir"]))
                        rows_out.append({"task": "dose_direction", "cond": f"x{a}", "arm": arm, "clip": ids[pick],
                                         "seed": np.full(n, seed), "cell": np.full(n, dd * 100 + ss),
                                         "achieved_frac": got / np.where(np.abs(intended) < 1e-6, np.nan, intended),
                                         "leak_spd": np.abs(r["spd"] - r0["spd"]),
                                         "leak_spd_mlp": np.abs(r["spd_mlp"] - r0["spd_mlp"]),
                                         "leak_pos": np.linalg.norm(r["pos"] - r0["pos"], axis=1),
                                         "dir_radius": r["dir_rad"]})
            evaluate("direction", pick, dl, tth, tv, real_mean, (dd, ss), extra=dose_dir)
        # ---------------- speed only (carriers in the target direction bin)
        pick = np.flatnonzero(test & (dbin == dd) & ~np.isin(sbin, hs))
        if len(pick):
            n = len(pick)
            ths, v_s = theta[pick], v[pick]
            tV = np.full(n, tv)
            dl = {"sheet": P(ths, tV) - P(ths, v_s), "speedline_global": line_delta(line_all, v_s, tV),
                  "speedline_dirwin": line_delta(line_dirwin(dd), v_s, tV),
                  "speedline_global_interp": line_delta(line_all_i, v_s, tV),
                  "sheet_tps_interp": P0(ths, tV) - P0(ths, v_s), "unedited": np.zeros((n, k))}

            def dose_spd(x, r0, dl=dl, n=n, v_s=v_s, pick=pick):
                if not dose:
                    return
                for arm in ("sheet", "speedline_global"):
                    for a in DOSES:
                        r = R.read(x + pca.lift_delta(a * dl[arm]))
                        intended = tv - v_s
                        rows_out.append({"task": "dose_speed", "cond": f"x{a}", "arm": arm, "clip": ids[pick],
                                         "seed": np.full(n, seed), "cell": np.full(n, dd * 100 + ss),
                                         "achieved_frac": (r["spd"] - r0["spd"]) / np.where(np.abs(intended) < 1e-6, np.nan, intended),
                                         "leak_dir": mf.value_error(r["dir"], r0["dir"], True),
                                         "leak_dir_mlp": mf.value_error(r["dir_mlp"], r0["dir_mlp"], True),
                                         "leak_pos": np.linalg.norm(r["pos"] - r0["pos"], axis=1)})
            evaluate("speed", pick, dl, tth, tv, real_mean, (dd, ss), extra=dose_spd)
    geo = {"lambda": lam, "smoothing": sm, "reinsch_lambda_global_ring": lam_r_all, "selection_table": sel, "held_dirs": hd, "held_speed_bins": hs,
           "n_knot_used": int(kf.sum()), "n_reader_clips": int((probe & ~held).sum()),
           "target_cells": [list(map(int, t)) for t in targets],
           "cone_radius_by_speed": {"v": np.linspace(v.min(), v.max(), 9).tolist(),
                                    "rho": frame.radius(np.linspace(v.min(), v.max(), 9)).tolist()}}
    return {"dataset": dataset, "layer": layer, "design": design, "seed": seed, "geo": geo, "rows": rows_out}


def extrapolate(dataset, layer, k=64):
    """Speed-only steer beyond the training range: all knot clips (no hold-out), carriers = test clips in speed bins
    4-6, targets above the max seen value (and one inside for reference)."""
    d = load_inputs(dataset, layer)
    theta = d["df"]["theta_degrees"].to_numpy(float)
    v = d["df"][COL[dataset]].to_numpy(float)
    vmid, vhalf = (v.min() + v.max()) / 2, (v.max() - v.min()) / 2
    dbin, sbin = bins(theta, v)
    knot, probe, test = (d["role"] == r for r in ("knot", "probe", "test"))
    pca = mf.fit_pca(d["X"][knot], k)
    Z = pca.project(d["X"])
    kc = cell_table(Z, theta, v, dbin, sbin, knot)
    P = Sheet(kc, 1.0, 0.1, vmid, vhalf)
    R = Readers(d, theta, v, probe, vmid, vhalf, 0)
    line_all = line_from(Z, knot, sbin, v)
    pick = np.flatnonzero(test & np.isin(sbin, [4, 5, 6]))
    ids = d["df"]["id"].to_numpy()
    x = d["X"][pick].astype(float)
    r0 = R.read(x)
    rows = []
    vmax = v.max()
    for tgt in (vmax - 0.0625 * vhalf * 2, vmax + 0.125 * vhalf, vmax + 0.25 * vhalf, vmax + 0.5 * vhalf):
        tV = np.full(len(pick), tgt)
        for arm, dz in (("sheet", P(theta[pick], tV) - P(theta[pick], v[pick])),
                        ("speedline_global", line_delta(line_all, v[pick], tV))):
            r = R.read(x + pca.lift_delta(dz))
            rows.append({"task": "extrapolate", "cond": f"v{tgt:.3f}", "arm": arm, "clip": ids[pick],
                         "seed": np.zeros(len(pick), int), "cell": np.zeros(len(pick), int),
                         "read_spd": r["spd"], "read_spd_mlp": r["spd_mlp"], "err_spd": np.abs(r["spd"] - tgt),
                         "leak_dir": mf.value_error(r["dir"], r0["dir"], True),
                         "leak_pos": np.linalg.norm(r["pos"] - r0["pos"], axis=1),
                         "edit_norm": np.linalg.norm(dz, axis=1)})
    return {"dataset": dataset, "layer": layer, "design": "extrapolate", "seed": 0, "rows": rows,
            "geo": {"vmax_seen": float(vmax)}}


def aggregate(outs, n_boot=1000):
    res = {}
    groups = {}
    for o in outs:
        for r in o["rows"]:
            key = (o["layer"], o["design"], r["task"], r["cond"])
            groups.setdefault(key, {}).setdefault(r["arm"], []).append(r)
    for (layer, design, task, cond), arms in groups.items():
        cat = {a: {q: np.concatenate([np.asarray(r[q]) for r in rs]) for q in rs[0] if q not in ("task", "cond", "arm")}
               for a, rs in arms.items()}
        metrics = [q for q in next(iter(cat.values())) if q not in ("clip", "seed", "cell")]
        node = res.setdefault(str(layer), {}).setdefault(design, {}).setdefault(task, {}).setdefault(cond, {})
        node["summary"] = {a: {q: vs.clip_boot(c[q], c["clip"], n_boot) for q in metrics if q in c} for a, c in cat.items()}
        if "sheet" in cat:
            node["gaps_sheet_minus"] = {}
            for a, c in cat.items():
                if a == "sheet" or len(c["clip"]) != len(cat["sheet"]["clip"]) or not (c["clip"] == cat["sheet"]["clip"]).all():
                    continue
                node["gaps_sheet_minus"][a] = {q: vs.clip_boot(cat["sheet"][q] - c[q], c["clip"], n_boot)
                                               for q in metrics if q in c and q in cat["sheet"]}
        # per-seed sheet gap vs the main 1-D baseline, and the block3 centre cell
        base = {"joint": "seq_local_dir_then_speed", "direction": "ring1d_band", "speed": "speedline_global"}.get(task)
        if base and base in cat and "sheet" in cat:
            q = {"joint": "err_dir", "direction": "err_dir", "speed": "err_spd"}[task]
            g = cat["sheet"][q] - cat[base][q]
            node["per_seed_gap_" + q + "_vs_" + base] = [float(g[cat["sheet"]["seed"] == s].mean())
                                                          for s in np.unique(cat["sheet"]["seed"])]
            node["per_cell_gap_" + q + "_vs_" + base] = {str(int(c_)): float(g[cat["sheet"]["cell"] == c_].mean())
                                                          for c_ in np.unique(cat["sheet"]["cell"])}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="speed", choices=tuple(COL))
    ap.add_argument("--layers", type=int, nargs="+", default=[12, 22])
    ap.add_argument("--designs", nargs="+", default=["block2", "block3", "cross"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--no-extrap", action="store_true")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "p5_velocity_sheet_v2.json"))
    ap.add_argument("--dump-rows", default=None, help="npz-free pickle of raw rows for figures")
    args = ap.parse_args()
    t0 = time.time()
    tasks = [delayed(run_seed)(args.dataset, l, de, s, 64, de == "block2") for l in args.layers for de in args.designs
             for s in args.seeds]
    if not args.no_extrap:
        tasks += [delayed(extrapolate)(args.dataset, l) for l in args.layers]
    outs = Parallel(n_jobs=args.jobs, verbose=5)(tasks)
    res = {"dataset": args.dataset, "results": aggregate(outs),
           "seeds": [{kk: o[kk] for kk in ("layer", "design", "seed", "geo")} for o in outs]}
    if args.dump_rows:
        import pickle
        Path(args.dump_rows).write_bytes(pickle.dumps([{kk: o[kk] for kk in ("layer", "design", "seed", "rows")} for o in outs]))
    res["provenance"] = provenance(seeds={"blocks": args.seeds, "bootstrap": 0, "mlp": "seed"}, layers=args.layers,
                                   designs=args.designs, dataset=args.dataset, k=64, pool="meanpool",
                                   wall_s=round(time.time() - t0, 1), cpu_only=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("wrote", args.out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

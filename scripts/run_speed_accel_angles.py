"""How speed and acceleration are represented and steered, beyond "they are straight lines" (REPORT section 4.1).

CPU only, on stored pooled activations. Roles from splits/split_v1.json via wm.p2_data.load_inputs: knot clips
(folds 0-2) build every line vector / edit, probe clips (folds 3-4) fit every readout, test clips are edited and
scored. Held-out (test) labels are never used in a fit. CIs: clip bootstrap (1000 draws) over the scored clips.

Line vector u (wm.speedaccel.line_vector): OLS slope of the raw features on the label over knot clips = mean
feature shift per unit label; an edit of dy units adds dy * u.

 A  rate vs displacement (speed set, timepool [N, 26, 8, D]): per-step line vectors U_t; U_t = b + tau_t d split
    into a step-constant (rate) part and a part growing with elapsed time (distance travelled s = v tau). Cosine of
    the speed line with the within-clip time direction (slope of the clip-mean-removed step features on t) and with
    the growth part d. Steering at points 12 / 22: add dv * u (all steps) or dv * U_t (step-specific) to test clips;
    read per-step probes of distance travelled, of t and of speed (fit on probe clips' step rows); a displacement
    code moves decoded distance by dv * tau_t (growing), a rate code by a constant.
 B  steering transfer around the ring (speed set, meanpool): per-45-degree-bin line vectors from knot clips; edit
    test clips in bin b with the vector of bin b + o (o = 0, 45, 90, 135, 180 deg), at the vector's own norm and at
    the global vector's norm; readouts: speed probe gain, direction-probe change, nearest-real R toward the mean of
    probe clips in the same bin at the target speed (+-0.3 m/s).
 C  metric and dose: (C1) edits of k x the unit step (1 m/s, 2.5 m/s^2), up and down, read by the probe (gain) and
    by density (5-NN distance to all real clips / real clips' own); (C2) knot spacing: centroid coordinate on the
    line vs label, linear vs log, and the cumulative-chord measure REPORT used, also on shuffled labels; (C3) real
    pixel twins beyond the training range (paper_layout -> clock_lam2: same path, speed x2; clock_acc_lam1 ->
    clock_acc_lam2: acceleration x4): how far the real twin moves along the line (gain), and nearest-real R of an
    edit toward the twin, in range vs out of range.
 D  acceleration vs speed: cos(u_speed, u_accel) with split-half reliability; probe transfer across sets (speed
    probe on acceleration clips: slope on a is an effective time; the clip's mean speed is a * 0.3125 s); edit
    cross-leak at matched norm; joint fit over both sets separating mean speed from acceleration; acceleration
    decodability with the speed line projected out.

  python scripts/run_speed_accel_angles.py --section A   (B, C, D; then --merge)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from wm import speedaccel as sa
from wm import timeman as tm
from wm.data import PROJECT_ROOT
from wm.p2_data import load_inputs
from wm.provenance import provenance

ACT = PROJECT_ROOT / "artifacts" / "activations"
STIM = PROJECT_ROOT / "artifacts" / "stimuli"
TMP = Path("/private/tmp/claude-501/-Users-stevenyang-Documents-world-mechanics-takehome/ae13a11f-788a-4ea6-a4fa-c5d4c3df1c0e/scratchpad/speed_accel_parts")
OUT = PROJECT_ROOT / "results" / "p5_speed_accel_angles.json"
FIG = PROJECT_ROOT / "figures" / "fig_speed_accel_angles.png"
TAU = tm.step_times()
TAU_BAR = float(TAU.mean())            # 0.3125 s: mean over the 8 steps of v(tau) = a tau for a clip from rest
POINTS = (4, 8, 12, 16, 19, 22, 25)
STEER_POINTS = (12, 22)
NB = 1000


def boot(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return tm.cluster_bootstrap(x, n_boot=NB) if len(x) > 1 else {"mean": float("nan"), "n": int(len(x))}


def data(dataset, L, model="vjepa2"):
    d = load_inputs(dataset, L, act_dir=ACT / dataset / model)
    d["X"] = np.nan_to_num(d["X"].astype(np.float64))
    return d


def probe(X, y):
    return tm.ridge(X, y)


def raw_weights(p):
    return p[-1].coef_ / p[0].scale_


def dir_probe(X, theta):
    t = np.radians(theta)
    return tm.ridge(X, np.column_stack([np.cos(t), np.sin(t)]))


def dir_read(p, X):
    P = p.predict(X)
    return np.degrees(np.arctan2(P[:, 1], P[:, 0])) % 360


def halves(ids):
    return np.asarray(ids) % 2 == 0


# ================================================================== A ============================================
def section_A():
    out = {}
    for L in POINTS:
        r = {}
        for ds, var in (("speed", "speed_mps"), ("acceleration", "acceleration_mps2")):
            d = load_inputs(ds, L)
            F = np.nan_to_num(np.asarray(np.load(ACT / ds / "vjepa2" / "timepool.npy", mmap_mode="r")[:, L], np.float64))
            y, knot = d["y"], d["role"] == "knot"
            h = halves(d["df"]["id"].to_numpy())
            M = F.mean(1)
            R = F - M[:, None]
            tt = np.broadcast_to(np.arange(8.0), (len(F), 8))

            def vecs(rows):
                u = sa.line_vector(M[rows], y[rows])
                U = np.stack([sa.line_vector(F[rows, t], y[rows]) for t in range(8)])
                tdir = sa.line_vector(R[rows].reshape(-1, F.shape[-1]), tt[rows].reshape(-1))
                rv = sa.rate_vs_displacement(U, TAU)
                tc = TAU - TAU.mean()
                dvec = tc @ U / (tc @ tc)
                return u, U, tdir, rv, dvec
            u, U, tdir, rv, dvec = vecs(knot)
            ua, Ua, ta, _, da = vecs(knot & h)
            ub, Ub, tb, _, db = vecs(knot & ~h)
            rel = {"u": sa.cos(ua, ub), "time_dir": sa.cos(ta, tb), "growth_d": sa.cos(da, db)}
            rng = np.random.default_rng(L)
            null = np.abs([sa.cos(tdir, rng.standard_normal(len(u))) for _ in range(1000)])
            r[ds] = {"rate_vs_displacement": rv,
                     "cos_U_t_with_u": [sa.cos(U[t], u) for t in range(8)],
                     "cos_u_time_dir": sa.cos(u, tdir),
                     "cos_u_time_dir_disattenuated": sa.disattenuated_cos(abs(sa.cos(u, tdir)), rel["u"], rel["time_dir"]),
                     "cos_growth_d_time_dir": sa.cos(dvec, tdir),
                     "cos_growth_d_time_dir_disattenuated": sa.disattenuated_cos(abs(sa.cos(dvec, tdir)), rel["growth_d"], rel["time_dir"]),
                     "split_half_reliability": rel, "random_dir_abs_cos_p95": float(np.percentile(null, 95)),
                     "norm_u_per_unit": float(np.linalg.norm(u)), "norm_time_dir_per_step": float(np.linalg.norm(tdir))}
            ws = None
            if ds == "speed" and L in STEER_POINTS:
                r[ds]["steer"], ws = steer_A(F, d, u, U)
            # knot-clip bootstrap of the line vectors (the edits' own uncertainty; readouts are fixed linear maps,
            # so a clip bootstrap of a constant edit has zero width by construction)
            kidx = np.flatnonzero(knot)
            brng = np.random.default_rng(100 + L)
            bs = {"rate_frac": [], "growth_frac": [], "abs_cos_u_time_dir": [], "ds_slope_step_specific": []}
            tc = TAU - TAU.mean()
            for _ in range(200):
                m = kidx[brng.integers(0, len(kidx), len(kidx))]
                ub_ = sa.line_vector(M[m], y[m])
                Ub_ = np.stack([sa.line_vector(F[m, t], y[m]) for t in range(8)])
                tb_ = sa.line_vector(R[m].reshape(-1, F.shape[-1]), tt[m].reshape(-1))
                rvb = sa.rate_vs_displacement(Ub_, TAU)
                bs["rate_frac"].append(rvb["rate_frac"]); bs["growth_frac"].append(rvb["growth_frac"])
                bs["abs_cos_u_time_dir"].append(abs(sa.cos(ub_, tb_)))
                if ws is not None:
                    dsb = Ub_ @ ws
                    bs["ds_slope_step_specific"].append(float(dsb @ tc / (tc @ tc)))
            r[ds]["knot_bootstrap_ci95"] = {k: np.percentile(v, [2.5, 97.5]).tolist() for k, v in bs.items() if v}
        out[str(L)] = r
        print("A", L, json.dumps({k: (v["rate_vs_displacement"]["rate_frac"], v["cos_u_time_dir"]) for k, v in r.items()}), flush=True)
    return out


def steer_A(F, d, u, U):
    y, role = d["y"], d["role"]
    pr, te = role == "probe", role == "test"
    D = F.shape[-1]
    s = tm.step_distance(y, np.zeros_like(y))
    rows = lambda m: F[m].reshape(-1, D)
    Ttr = np.broadcast_to(np.arange(8.0), (pr.sum(), 8)).reshape(-1)
    p_s = probe(rows(pr), s[pr].reshape(-1))
    p_t = probe(rows(pr), Ttr)
    p_v = probe(rows(pr), np.repeat(y[pr], 8))
    dv = np.where(y[te] < 2.125, 1.0, -1.0)
    F0 = F[te]
    edits = {"constant_u": F0 + dv[:, None, None] * u[None, None],
             "step_specific_U_t": F0 + dv[:, None, None] * U[None]}
    base = {k: p.predict(F0.reshape(-1, D)).reshape(-1, 8) for k, p in (("s", p_s), ("t", p_t), ("v", p_v))}
    out = {"probe_test_r2": {"distance": tm.r2(s[te].reshape(-1), base["s"].reshape(-1)),
                             "t": tm.r2(np.broadcast_to(np.arange(8.0), (te.sum(), 8)).reshape(-1), base["t"].reshape(-1)),
                             "speed": tm.r2(np.repeat(y[te], 8), base["v"].reshape(-1))},
           "kinematic_prediction_ds_per_mps": TAU.tolist()}
    for name, Fe in edits.items():
        pe = {k: p.predict(Fe.reshape(-1, D)).reshape(-1, 8) for k, p in (("s", p_s), ("t", p_t), ("v", p_v))}
        ds_ = (pe["s"] - base["s"]) / dv[:, None]        # per m/s of intended change
        dt_ = (pe["t"] - base["t"]) / dv[:, None]
        dv_ = ((pe["v"] - base["v"]) / dv[:, None]).mean(1)
        # per clip: slope of decoded distance change on tau (kinematic = 1: ds_t = dv tau_t) and its intercept
        tc = TAU - TAU.mean()
        slope = ds_ @ tc / (tc @ tc)
        icpt = ds_.mean(1) - slope * TAU.mean()
        out[name] = {"speed_gain": boot(dv_), "ds_per_step_mean": ds_.mean(0).tolist(),
                     "ds_slope_on_tau": boot(slope), "ds_intercept_m": boot(icpt),
                     "ds_mean_over_steps_over_kinematic": boot(ds_.mean(1) / TAU.mean()),
                     "dt_steps_mean": boot(dt_.mean(1))}
    return out, raw_weights(p_s)


# ================================================================== B ============================================
def section_B():
    out = {}
    nb = 8
    for L in STEER_POINTS:
        d = data("speed", L)
        X, y, th, role = d["X"], d["y"], d["df"]["theta_degrees"].to_numpy(float), d["role"]
        knot, pr, te = role == "knot", role == "probe", role == "test"
        b = sa.angle_bins(th, nb)
        h = halves(d["df"]["id"].to_numpy())
        ug = sa.line_vector(X[knot], y[knot])
        ub = {k: sa.line_vector(X[knot & (b == k)], y[knot & (b == k)]) for k in range(nb)}
        uh = {k: (sa.line_vector(X[knot & (b == k) & h], y[knot & (b == k) & h]),
                  sa.line_vector(X[knot & (b == k) & ~h], y[knot & (b == k) & ~h])) for k in range(nb)}
        rel_bin = float(np.mean([sa.cos(*uh[k]) for k in range(nb)]))
        cos_by_off = {o: float(np.mean([sa.cos(ub[k], ub[(k + o) % nb]) for k in range(nb)])) for o in range(1, 5)}
        # the same between two independent halves of the SAME bin is the noise ceiling (rel_bin)
        p_v, p_d = probe(X[pr], y[pr]), dir_probe(X[pr], th[pr])
        idx = np.flatnonzero(te)
        dv = np.where(y[idx] < 2.125, 1.5, -1.5)
        tgt = y[idx] + dv
        v0, d0 = p_v.predict(X[idx]), dir_read(p_d, X[idx])
        # nearest-real target: probe clips in the same bin within 0.3 m/s of the target
        T = np.full((len(idx), X.shape[1]), np.nan)
        for i, (k, t_) in enumerate(zip(b[idx], tgt)):
            m = pr & (b == k) & (np.abs(y - t_) <= 0.3)
            if m.sum() >= 3:
                T[i] = X[m].mean(0)
        okT = np.isfinite(T).all(1)
        rng = np.random.default_rng(L)
        rnd = sa.unit(rng.standard_normal(X.shape[1])) * np.linalg.norm(ug)

        def score(vec_of_clip):
            V = np.stack(vec_of_clip)
            Xe = X[idx] + dv[:, None] * V
            g = (p_v.predict(Xe) - v0) / dv
            dth = np.abs(sa.circ_diff_deg(dir_read(p_d, Xe), d0))
            R = np.full(len(idx), np.nan)
            R[okT] = sa.nearest_real_R(Xe[okT], X[idx][okT], T[okT])
            return {"speed_gain": boot(g), "dir_change_deg": boot(dth), "nearest_real_R": boot(R), "_R": R}
        r = {"n_test": int(len(idx)), "n_with_real_target": int(okT.sum()),
             "cos_bin_vectors_by_offset": {str(45 * o): c for o, c in cos_by_off.items()},
             "cos_bin_vs_global_mean": float(np.mean([sa.cos(ub[k], ug) for k in range(nb)])),
             "split_half_reliability_bin_vector": rel_bin,
             "norm_ratio_bin_over_global": [float(np.linalg.norm(ub[k]) / np.linalg.norm(ug)) for k in range(nb)],
             "global": score([ug] * len(idx)), "random_dir_global_norm": score([rnd] * len(idx))}
        for o in range(0, 5):
            own = [ub[(k + o) % nb] for k in b[idx]]
            if o in (1, 2, 3):              # +o and -o bins scored separately
                sp = score([ub[(k + o) % nb] for k in b[idx]])
                sm = score([ub[(k - o) % nb] for k in b[idx]])
                spn = score([sa.unit(ub[(k + o) % nb]) * np.linalg.norm(ug) for k in b[idx]])
                smn = score([sa.unit(ub[(k - o) % nb]) * np.linalg.norm(ug) for k in b[idx]])
                r[f"offset_{45 * o}"] = {"own_norm": {"plus": sp, "minus": sm}, "global_norm": {"plus": spn, "minus": smn}}
            else:
                r[f"offset_{45 * o}"] = {"own_norm": score(own),
                                         "global_norm": score([sa.unit(v) * np.linalg.norm(ug) for v in own])}
        Rv = lambda k, nm="global_norm", sd=None: (r[k][nm]["_R"] if sd is None else 0.5 * (r[k][nm]["plus"]["_R"] + r[k][nm]["minus"]["_R"]))
        own = r["offset_0"]["global_norm"]["_R"]
        pairs = {"own_minus_90": own - Rv("offset_90", sd=1), "own_minus_135": own - Rv("offset_135", sd=1),
                 "own_minus_180": own - r["offset_180"]["global_norm"]["_R"],
                 "180_minus_135": r["offset_180"]["global_norm"]["_R"] - Rv("offset_135", sd=1),
                 "own_minus_global": own - r["global"]["_R"],
                 "own_minus_90_own_norm": r["offset_0"]["own_norm"]["_R"] - Rv("offset_90", "own_norm", 1)}
        r["paired_nearest_real_R_diff"] = {k: boot(v) for k, v in pairs.items()}

        def strip(x):
            if isinstance(x, dict):
                return {k: strip(v) for k, v in x.items() if k != "_R"}
            return x
        r = strip(r)
        out[str(L)] = r
        print("B", L, {k: round(v["mean"], 3) for k, v in r["paired_nearest_real_R_diff"].items()}, {k: (v["own_norm"]["speed_gain"]["mean"] if "mean" in v.get("own_norm", {}).get("speed_gain", {}) else
                            v["own_norm"]["plus"]["speed_gain"]["mean"]) for k, v in r.items() if k.startswith("offset")}, flush=True)
    return out


# ================================================================== C ============================================
KS = (0.25, 0.5, 1, 2, 4, 8)
UNIT = {"speed": 1.0, "acceleration": 2.5}
RANGE = {"speed": (0.25, 4.0), "acceleration": (0.25, 10.0)}


def section_C():
    out = {"dose": {}, "spacing": {}, "twins": {}}
    for L in STEER_POINTS:
        for ds in ("speed", "acceleration"):
            for model in ("vjepa2", "random"):
                d = data(ds, L, model)
                X, y, role = d["X"], d["y"], d["role"]
                knot, pr, te = role == "knot", role == "probe", role == "test"
                u = sa.line_vector(X[knot], y[knot])
                p = probe(X[pr], y[pr])
                idx = np.flatnonzero(te)
                y0, p0 = y[idx], p.predict(X[idx])
                # nonparametric readout: mean label of the 10 nearest probe clips (bounded by the data's range)
                Xp, yp = X[pr].astype(np.float32), y[pr]
                sqp = (Xp ** 2).sum(1)

                def knn_label(Q, k=10):
                    Q = Q.astype(np.float32)
                    d2 = (Q ** 2).sum(1)[:, None] + sqp[None] - 2 * Q @ Xp.T
                    return yp[np.argsort(d2, 1)[:, :k]].mean(1)
                n0 = knn_label(X[idx])
                res = {"probe_test_r2": tm.r2(y0, p0), "knn_label_test_r2": tm.r2(y0, n0),
                       "note": "probe gain is constant in k by linearity (linear probe, linear edit); knn-label and density are the dose readouts",
                       "knn_ratio_unedited": boot(sa.knn_ratio(X[idx], X[~te], k=5)),
                       "knn_reference": "train clips only (knot + probe), so no test clip is its own neighbour"}
                lo, hi = RANGE[ds]
                for k in KS:
                    for sgn, nm in ((1, "up"), (-1, "down")):
                        dy = sgn * k * UNIT[ds]
                        Xe = X[idx] + dy * u
                        g = (p.predict(Xe) - p0) / dy
                        kr = sa.knn_ratio(Xe, X[~te], k=5)
                        gk = (knn_label(Xe) - n0) / dy
                        inr = (y0 + dy >= lo) & (y0 + dy <= hi)
                        res[f"k{k}_{nm}"] = {"dy": dy, "n_in": int(inr.sum()), "n_out": int((~inr).sum()),
                                             "gain_in": boot(g[inr]), "gain_out": boot(g[~inr]),
                                             "knn_label_gain_in": boot(gk[inr]), "knn_label_gain_out": boot(gk[~inr]),
                                             "knn_in": boot(kr[inr]), "knn_out": boot(kr[~inr])}
                out["dose"][f"{ds}/{model}/{L}"] = res
                if model == "vjepa2":
                    out["spacing"][f"{ds}/{L}"] = spacing(X[knot], y[knot], u)
                print("C", ds, model, L, {k: (round(v["knn_label_gain_in"].get("mean", np.nan), 3), round(v["knn_label_gain_out"].get("mean", np.nan), 3), round(v["knn_out"].get("mean", np.nan), 2)) for k, v in res.items() if k.endswith("_up")}, flush=True)
        for fam, (a_set, b_set, var, prim) in {"velocity": ("paper_layout", "clock_lam2", "speed_mps", "speed"),
                                               "acceleration": ("clock_acc_lam1", "clock_acc_lam2", "acceleration_mps2", "acceleration")}.items():
            out["twins"][f"{fam}/{L}"] = twins(L, a_set, b_set, var, prim)
    return out


def spacing(Xk, yk, u):
    vals = np.unique(yk)
    C = np.stack([Xk[yk == v].mean(0) for v in vals])
    coord = (C - C.mean(0)) @ sa.unit(u)
    rng = np.random.default_rng(0)
    ysh = rng.permutation(yk)
    Csh = np.stack([Xk[ysh == v].mean(0) for v in vals])
    pc1 = np.linalg.svd(C - C.mean(0), full_matrices=False)[2][0]
    return {"line_coordinate": sa.spacing_r2(coord, vals),
            "centroid_pc1_coordinate_label_free": sa.spacing_r2((C - C.mean(0)) @ pc1, vals),
            "cumulative_chord_real": sa.spacing_r2(sa.cumulative_chord(C)[1:], vals[1:]),
            "cumulative_chord_shuffled_labels": sa.spacing_r2(sa.cumulative_chord(Csh)[1:], vals[1:]),
            "n_values": int(len(vals)), **spacing_boot(Xk, yk)}


def spacing_boot(Xk, yk, n=200):
    """Knot-clip bootstrap of r2_log - r2_linear for the label-free centroid-PC1 coordinate, over all values and
    over values >= 0.75 (drops the slowest / weakest-acceleration clips, which could drive a log fit alone)."""
    rng = np.random.default_rng(1)
    out = {}
    for nm, keep in (("all_values", yk > 0), ("values_ge_0.75", yk >= 0.75)):
        X_, y_ = Xk[keep], yk[keep]
        vals = np.unique(y_)
        diffs = []
        for _ in range(n):
            i = rng.integers(0, len(y_), len(y_))
            Xb, yb = X_[i], y_[i]
            vb = np.array([v for v in vals if (yb == v).any()])
            C = np.stack([Xb[yb == v].mean(0) for v in vb])
            pc1 = np.linalg.svd(C - C.mean(0), full_matrices=False)[2][0]
            r = sa.spacing_r2((C - C.mean(0)) @ pc1, vb)
            diffs.append(r["r2_log"] - r["r2_linear"])
        C = np.stack([X_[y_ == v].mean(0) for v in vals])
        pc1 = np.linalg.svd(C - C.mean(0), full_matrices=False)[2][0]
        r = sa.spacing_r2((C - C.mean(0)) @ pc1, vals)
        out[f"pc1_{nm}"] = {**r, "log_minus_linear": r["r2_log"] - r["r2_linear"],
                            "log_minus_linear_ci95": np.percentile(diffs, [2.5, 97.5]).tolist()}
    return out


def stim_meta(name):
    ids = json.loads((ACT / f"stimuli_{name}" / "vjepa2" / "ids.json").read_text())
    rows = [json.loads((STIM / name / "videos" / f"scene_{int(i):04d}" / "metadata.json").read_text()) for i in ids]
    if name == "paper_layout":          # lam = 1 reference: in-frame flags recomputed as in run_clock_test.py
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from render_clock_stimuli import clock_meta
        rows = [clock_meta(m, 1.0, name) for m in rows]
    return ids, rows


def twins(L, a_set, b_set, var, prim):
    d = data(prim, L)
    X, y, role = d["X"], d["y"], d["role"]
    knot, pr = role == "knot", role == "probe"
    u = sa.line_vector(X[knot], y[knot])
    p = probe(X[pr], y[pr])
    ida, ma = stim_meta(a_set)
    idb, mb = stim_meta(b_set)
    Xa = np.asarray(np.load(ACT / f"stimuli_{a_set}" / "vjepa2" / "meanpool.npy", mmap_mode="r")[:, L], np.float64)
    Xb = np.asarray(np.load(ACT / f"stimuli_{b_set}" / "vjepa2" / "meanpool.npy", mmap_mode="r")[:, L], np.float64)
    src = {int(m.get("source_id", m["id"])): j for j, m in enumerate(mb)}
    motion = "velocity" if var == "speed_mps" else "acceleration"
    I, J = [], []
    for i, m in enumerate(ma):
        j = src.get(int(m.get("source_id", m["id"])))
        if (j is not None and m["motion"] == motion and mb[j]["motion"] == motion and m["in_frame_all_frames"]
                and mb[j]["in_frame_all_frames"] and m[var] > 0):
            I.append(i), J.append(j)
    I, J = np.array(I), np.array(J)
    ya, yb = np.array([ma[i][var] for i in I]), np.array([mb[j][var] for j in J])
    A, B = Xa[I], Xb[J]
    dy = yb - ya
    gain, frac = sa.along_line_gain(B - A, u, dy)
    pa, pb = p.predict(A), p.predict(B)
    hi = RANGE[prim][1]
    out = {"n_pairs": int(len(I)), "source_values": sorted(set(ya.tolist())), "target_values": sorted(set(yb.tolist())),
           "train_range": RANGE[prim]}
    for nm, m in (("target_in_range", yb <= hi), ("target_out_of_range", yb > hi)):
        if m.sum() < 2:
            continue
        r = {"n": int(m.sum()), "twin_along_line_gain": boot(gain[m]), "twin_frac_energy_on_line": boot(frac[m]),
             "twin_probe_gain": boot((pb[m] - pa[m]) / dy[m])}
        for k in (0.5, 1.0, 2.0):
            E = A[m] + (k * dy[m])[:, None] * u
            r[f"edit_k{k}"] = {"nearest_real_R_to_twin": boot(sa.nearest_real_R(E, A[m], B[m])),
                               "probe_gain": boot((p.predict(E) - pa[m]) / dy[m])}
        # per target value: along-line gain of the real twin (does the representation saturate past the range?)
        r["gain_by_target"] = {str(v): float(np.mean(gain[m & (yb == v)])) for v in sorted(set(yb[m].tolist()))}
        r["gain_by_target_boot"] = {str(v): boot(gain[m & (yb == v)]) for v in sorted(set(yb[m].tolist()))}
        r["move_along_line_by_target_in_units_of_u"] = {str(v): float(np.mean((gain * dy)[m & (yb == v)]))
                                                        for v in sorted(set(yb[m].tolist()))}
        out[nm] = r
    print("C3", var, L, {k: out[k]["twin_along_line_gain"]["mean"] for k in ("target_in_range", "target_out_of_range") if k in out}, flush=True)
    return out


# ================================================================== D ============================================
def section_D():
    out = {}
    for model in ("vjepa2", "random"):
        for L in POINTS:
            S, A = data("speed", L, model), data("acceleration", L, model)
            ks, ka = S["role"] == "knot", A["role"] == "knot"
            hs, ha = halves(S["df"]["id"].to_numpy()), halves(A["df"]["id"].to_numpy())
            uv, ua = sa.line_vector(S["X"][ks], S["y"][ks]), sa.line_vector(A["X"][ka], A["y"][ka])
            rel_v = sa.cos(sa.line_vector(S["X"][ks & hs], S["y"][ks & hs]), sa.line_vector(S["X"][ks & ~hs], S["y"][ks & ~hs]))
            rel_a = sa.cos(sa.line_vector(A["X"][ka & ha], A["y"][ka & ha]), sa.line_vector(A["X"][ka & ~ha], A["y"][ka & ~ha]))
            th = np.radians(S["df"]["theta_degrees"].to_numpy(float))
            ucos = sa.line_vector(S["X"][ks], np.cos(th)[ks])
            c = sa.cos(uv, ua)
            r = {"cos_speed_accel_line": c, "reliability_speed": rel_v, "reliability_accel": rel_a,
                 "cos_disattenuated": sa.disattenuated_cos(c, rel_v, rel_a),
                 "cos_speed_line_vs_cos_theta_line": sa.cos(uv, ucos)}
            ps, pa = probe(S["X"][S["role"] == "probe"], S["y"][S["role"] == "probe"]), probe(A["X"][A["role"] == "probe"], A["y"][A["role"] == "probe"])
            r["cos_probe_weights"] = sa.cos(raw_weights(ps), raw_weights(pa))
            ts, ta = S["role"] == "test", A["role"] == "test"
            r["probe_test_r2"] = {"speed": tm.r2(S["y"][ts], ps.predict(S["X"][ts])), "accel": tm.r2(A["y"][ta], pa.predict(A["X"][ta]))}
            # transfer: speed probe on acceleration test clips; accel probe on speed test clips
            vh = ps.predict(A["X"][ta])
            ah = pa.predict(S["X"][ts])
            sl, ic, r2 = sa.ols_slope(A["y"][ta], vh)
            sl2, ic2, r22 = sa.ols_slope(S["y"][ts], ah)
            bs = [sa.ols_slope(A["y"][ta][i], vh[i])[0] for i in np.random.default_rng(0).integers(0, ta.sum(), (NB, ta.sum()))]
            bs2 = [sa.ols_slope(S["y"][ts][i], ah[i])[0] for i in np.random.default_rng(1).integers(0, ts.sum(), (NB, ts.sum()))]
            r["transfer"] = {"speed_probe_on_accel_clips": {"slope_mps_per_mps2": sl, "ci95": np.percentile(bs, [2.5, 97.5]).tolist(),
                                                           "intercept": ic, "r2_vs_a": r2,
                                                           "kinematic_mean_speed_slope": TAU_BAR, "kinematic_final_speed_slope": float(TAU[-1]),
                                                           "mae_vs_mean_speed": float(np.abs(vh - TAU_BAR * A["y"][ta]).mean())},
                             "accel_probe_on_speed_clips": {"slope_mps2_per_mps": sl2, "ci95": np.percentile(bs2, [2.5, 97.5]).tolist(),
                                                           "intercept": ic2, "r2_vs_v": r22, "kinematic_if_mean_speed_code": 1 / TAU_BAR}}
            # edits and cross-leak (test clips, toward the middle of the range)
            dv = np.where(S["y"][ts] < 2.125, 1.0, -1.0)
            da = np.where(A["y"][ta] < 5.125, 2.5, -2.5)
            XS, XA = S["X"][ts], A["X"][ta]
            rnd = sa.unit(np.random.default_rng(L).standard_normal(len(uv)))
            vs0, as0, va0, aa0 = ps.predict(XS), pa.predict(XS), ps.predict(XA), pa.predict(XA)
            nv, na = np.linalg.norm(uv), np.linalg.norm(ua)
            ed = {}
            for nm, vec in (("speed_line", uv), ("accel_dir_at_speed_norm", sa.unit(ua) * nv), ("random_at_speed_norm", rnd * nv)):
                E = XS + dv[:, None] * vec
                ed[f"speed_clips/{nm}"] = {"speed_probe_per_mps": boot((ps.predict(E) - vs0) / dv),
                                           "accel_probe_per_mps": boot((pa.predict(E) - as0) / dv)}
            for nm, vec in (("accel_line", ua), ("speed_dir_at_accel_norm", sa.unit(uv) * na), ("random_at_accel_norm", rnd * na)):
                E = XA + da[:, None] * vec
                ed[f"accel_clips/{nm}"] = {"accel_probe_per_mps2": boot((pa.predict(E) - aa0) / da),
                                           "speed_probe_per_mps2": boot((ps.predict(E) - va0) / da)}
            r["edits"] = ed
            r["edits_note"] = ("speed_probe_per_mps2 under the accel line vs 0.3125 s = the mean-speed change a real clip "
                               "gets per m/s^2; accel_probe_per_mps under the speed line vs 3.2 = 1/0.3125 if the "
                               "acceleration code were a mean-speed code")
            # joint fit: mean speed and acceleration separated (speed set has a = 0)
            Xj = np.vstack([S["X"][ks], A["X"][ka]])
            vbar = np.concatenate([S["y"][ks], TAU_BAR * A["y"][ka]])
            acc = np.concatenate([np.zeros(ks.sum()), A["y"][ka]])
            isa = np.concatenate([np.zeros(ks.sum()), np.ones(ka.sum())])
            Bj = sa.joint_line_vectors(Xj, np.column_stack([vbar, acc, isa]))
            bv, ba = Bj[0], Bj[1]
            r["joint"] = {"cos_pure_accel_vs_speed_line": sa.cos(ba, uv), "cos_meanspeed_vs_speed_line": sa.cos(bv, uv),
                          "cos_accel_line_vs_meanspeed_part": sa.cos(ua, TAU_BAR * bv),
                          "norm_meanspeed_part_over_accel_line": float(np.linalg.norm(TAU_BAR * bv) / na),
                          "norm_pure_accel_over_accel_line": float(np.linalg.norm(ba) / na),
                          "frac_accel_line_energy_along_speed_line": float(sa.cos(ua, uv) ** 2)}
            # acceleration decodability with the speed line (and a 4-D speed subspace) projected out
            vals = np.unique(S["y"][ks])
            Cs = np.stack([S["X"][ks][S["y"][ks] == v].mean(0) for v in vals])
            Vt = np.linalg.svd(Cs - Cs.mean(0), full_matrices=False)[2]
            dec = {}
            for nm, Q in (("none", np.zeros((0, len(uv)))), ("speed_line", sa.unit(uv)[None]), ("speed_centroid_pcs4", Vt[:4])):
                P = lambda Z: Z - (Z @ Q.T) @ Q
                pq = probe(P(A["X"][A["role"] == "probe"]), A["y"][A["role"] == "probe"])
                dec[nm] = tm.r2(A["y"][ta], pq.predict(P(XA)))
            r["accel_r2_after_projecting_out"] = dec
            r["accel_r2_after_projecting_out_note"] = ("weak test: removing one direction does not remove a redundant "
                                                       "code; see matched_mean_speed for the decisive one")
            r["matched_mean_speed"] = matched_mean_speed(S, A)
            out[f"{model}/{L}"] = r
            print("D", model, L, round(c, 3), round(r["cos_disattenuated"], 3), round(sl, 3), round(sl2, 3), dec, flush=True)
    return out


def matched_mean_speed(S, A, lo=0.4, hi=3.0):
    """Can a linear readout tell an accelerating clip from a constant-velocity clip of the same mean speed?
    Logistic regression (standardised, CV-chosen C) on probe clips of both sets, scored on test clips of both
    sets whose mean speed (v, or a * 0.3125 s) lies in [lo, hi]. If the acceleration code were only a mean-speed
    code, AUROC would be ~0.5. Also: a ridge probe of a (0 for the speed set) fit on probe clips of both sets,
    its test R^2 over both sets, and its mean reading on speed-set test clips."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    mv = lambda D, acc: (TAU_BAR * D["y"]) if acc else D["y"]
    Xp = np.vstack([S["X"][S["role"] == "probe"], A["X"][A["role"] == "probe"]])
    lp = np.concatenate([np.zeros((S["role"] == "probe").sum()), np.ones((A["role"] == "probe").sum())])
    ap = np.concatenate([np.zeros((S["role"] == "probe").sum()), A["y"][A["role"] == "probe"]])
    ts, ta = S["role"] == "test", A["role"] == "test"
    ms, ma = ts & (mv(S, 0) >= lo) & (mv(S, 0) <= hi), ta & (mv(A, 1) >= lo) & (mv(A, 1) <= hi)
    Xt = np.vstack([S["X"][ms], A["X"][ma]])
    lt = np.concatenate([np.zeros(ms.sum()), np.ones(ma.sum())])
    clf = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-4, 1, 6), cv=5, max_iter=2000)).fit(Xp, lp)
    sc = clf.decision_function(Xt)
    rng = np.random.default_rng(0)
    aucs = []
    for _ in range(NB):
        i = rng.integers(0, len(lt), len(lt))
        if 0 < lt[i].sum() < len(i):
            aucs.append(roc_auc_score(lt[i], sc[i]))
    pa = tm.ridge(Xp, ap)
    Xall = np.vstack([S["X"][ts], A["X"][ta]])
    aall = np.concatenate([np.zeros(ts.sum()), A["y"][ta]])
    pr = pa.predict(Xall)
    return {"auroc_accel_vs_const_matched_mean_speed": float(roc_auc_score(lt, sc)),
            "auroc_ci95": np.percentile(aucs, [2.5, 97.5]).tolist(), "n_speed": int(ms.sum()), "n_accel": int(ma.sum()),
            "mean_speed_window": [lo, hi],
            "pooled_accel_probe_test_r2": tm.r2(aall, pr),
            "pooled_accel_probe_mean_on_speed_clips": float(pr[:ts.sum()].mean()),
            "pooled_accel_probe_accel_clips_slope": sa.ols_slope(A["y"][ta], pr[ts.sum():])[0]}


# ================================================================== merge / figure ================================
def merge():
    parts = {s: json.loads((TMP / f"{s}.json").read_text()) for s in "ABCD" if (TMP / f"{s}.json").exists()}
    res = {"A_rate_vs_displacement": parts.get("A"), "B_ring_transfer": parts.get("B"),
           "C_metric_dose_extrapolation": parts.get("C"), "D_accel_vs_speed": parts.get("D"),
           "provenance": provenance(script="scripts/run_speed_accel_angles.py", roles="knot folds 0-2 build, probe folds 3-4 read, test clips edited",
                                    tau_bar_s=TAU_BAR, points=list(POINTS), steer_points=list(STEER_POINTS), n_boot=NB,
                                    parts_timestamps={s: parts[s].get("_ts") for s in parts})}
    OUT.write_text(json.dumps(res, indent=1))
    figure(res)
    print("wrote", OUT, FIG)


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    A, B, C, D = (res[k] for k in ("A_rate_vs_displacement", "B_ring_transfer", "C_metric_dose_extrapolation", "D_accel_vs_speed"))
    if A:
        P = sorted(int(k) for k in A if k.isdigit())
        for ds, col in (("speed", "#2a6fdb"), ("acceleration", "#c0392b")):
            ax[0].plot(P, [A[str(p)][ds]["rate_vs_displacement"]["rate_frac"] for p in P], "-o", color=col, label=f"{ds}: step-constant share")
            ax[0].plot(P, [abs(A[str(p)][ds]["cos_u_time_dir"]) for p in P], "--s", color=col, label=f"{ds}: |cos(line, time dir)|")
        ax[0].plot(P, [A[str(p)]["speed"]["random_dir_abs_cos_p95"] for p in P], ":", color="#888", label="random |cos| p95")
        ax[0].set_xlabel("point"); ax[0].set_ylim(0, 1.05); ax[0].legend(fontsize=6, frameon=False)
        ax[0].set_title("A: rate code, orthogonal to time", fontsize=9)
    if B:
        for L, mk in (("12", "o"), ("22", "s")):
            r = B[L]
            offs = [0, 45, 90, 135, 180]
            g = []
            for o in offs:
                x = r[f"offset_{o}"]["global_norm"]
                q = "nearest_real_R"
                g.append(x[q]["mean"] if q in x else 0.5 * (x["plus"][q]["mean"] + x["minus"][q]["mean"]))
            col = "C0" if L == "12" else "C1"
            ax[1].plot(offs, g, "-" + mk, color=col, label=f"point {L}: bin vector (global norm)")
            ax[1].axhline(r["global"]["nearest_real_R"]["mean"], ls="--", lw=0.8, color=col, label=f"point {L}: global vector")
        ax[1].axhline(0, color="#888", lw=0.6)
        ax[1].set_xlabel("vector's bin offset from clip's bin (deg)"); ax[1].set_ylabel("nearest-real R (same-bin real clips)")
        ax[1].legend(fontsize=6, frameon=False); ax[1].set_title("B: speed edit transfer around the ring (probe gain flat 0.98)", fontsize=9)
    if C:
        for key, col in (("velocity/22", "#2a6fdb"), ("acceleration/22", "#c0392b"), ("velocity/12", "#9bbcf0"), ("acceleration/12", "#f1a9a0")):
            t = C["twins"].get(key, {})
            xs, ys = [], []
            for nm in ("target_in_range", "target_out_of_range"):
                for v, gv in t.get(nm, {}).get("gain_by_target", {}).items():
                    xs.append(float(v)); ys.append(gv)
            o = np.argsort(xs)
            ax[2].plot(np.array(xs)[o], np.array(ys)[o], "-o", ms=3, color=col, label=key)
        ax[2].axhline(1, color="#888", lw=0.8)
        ax[2].set_xlabel("twin's label value (m/s or m/s^2)"); ax[2].set_ylabel("real twin's move along line / linear")
        ax[2].legend(fontsize=6, frameon=False); ax[2].set_title("C: real clips beyond the range", fontsize=9)
    if D:
        for model, ls in (("vjepa2", "-o"), ("random", ":s")):
            P = sorted(int(k.split("/")[1]) for k in D if k.startswith(model))
            ax[3].plot(P, [D[f"{model}/{p}"]["cos_speed_accel_line"] for p in P], ls, color="#2a6fdb", label=f"{model} cos(speed, accel line)")
            ax[3].plot(P, [D[f"{model}/{p}"]["transfer"]["speed_probe_on_accel_clips"]["slope_mps_per_mps2"] for p in P], ls, color="#c0392b",
                       label=f"{model} speed probe slope on a (s)")
            ax[3].plot(P, [D[f"{model}/{p}"]["matched_mean_speed"]["auroc_accel_vs_const_matched_mean_speed"] for p in P], ls,
                       color="#27ae60", label=f"{model} AUROC accel vs const, matched mean speed")
        ax[3].axhline(TAU_BAR, color="#888", lw=0.8, ls="--")
        ax[3].set_xlabel("point"); ax[3].legend(fontsize=6, frameon=False); ax[3].set_title("D: acceleration vs speed", fontsize=9)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", choices=list("ABCD"))
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge()
        sys.exit()
    from datetime import datetime, timezone
    TMP.mkdir(parents=True, exist_ok=True)
    res = {"A": section_A, "B": section_B, "C": section_C, "D": section_D}[a.section]()
    res["_ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (TMP / f"{a.section}.json").write_text(json.dumps(res))

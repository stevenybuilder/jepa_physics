"""Part 2 extension: the TRUE energy geodesic of Goodfire 2605.05115 section 3.4 (Eq. 4 length under the Eq. 6 density
metric G_E(h) = (alpha e^{-E(h)} + beta)^{-1} I), against the spline (Eq. 2 proxy) and the raw chord, on the held-out
direction arcs of results/p2_steer_direction_direction_L{12,22}_contiguous_rawchord.json.

Same edit space, split, targets, carriers, K = 50 readouts and evaluation code as that file (run_part2.build /
pick_clips / evaluate, spline=smooth, holdout=contiguous, PCA-64, additive, residual kept). Per carrier the path runs
from its own PCA-64 state to the RAW CHORD arm's end state (Z + raw(tgt) - raw(src)), so the chord and every
geodesic share both endpoints. 50 free nodes between the fixed endpoints; discrete Eq. 4 length with Simpson
quadrature of sqrt(g) per segment; torch L-BFGS (strong Wolfe) from session2_pullback_goodfire.optimise.

Energies (TRAIN = knot clips, folds 0-2 at kept values only; the held-out arc, probe folds and test never enter):
  knn  E = d_int * log rbar_5(h): rbar_5 = mean full-space distance to the 5 nearest train clips (off_manifold_energy's
       knn=5 measure, with the carrier's fixed off-subspace residual), d_int = Levina-Bickel MLE intrinsic dim (k=10)
       -- the kNN density estimate p ~ rbar^{-d}.
  kde  E = -log sum_j exp(-|u - u_j|^2 / 2h^2), u = top-m PCs whitened by train SD; h by 3-fold (fold 0/1/2)
       held-out log-likelihood.
Calibration (Bethune et al. 2025, cited by Goodfire for alpha, beta; Goodfire gives no values): p~ = exp(-(E - E_ref)),
E_ref = median leave-one-out train energy, alpha = 1 - beta, beta = 1e-3, so G = I at a typical train point and
G -> 1e3 I where the density vanishes.

  python scripts/run_geodesic.py --layer 12 [--arcs 1-16] [--n-clips 48]
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rp2 = _load("rp2", "run_part2.py")
pbg = _load("pbg", "session2_pullback_goodfire.py")
from wm import manifold as mf  # noqa: E402
from wm.p2_data import load_inputs  # noqa: E402
from wm.provenance import provenance  # noqa: E402

torch.set_default_dtype(torch.float64)
BETA = 1e-3                     # Bethune et al.: metric ~ I on data, ~ 1e3 I in low density
N_FREE = 50                     # free nodes between the fixed endpoints
KNN_K, DIM_K = 5, 10
LBFGS_CFG = {**pbg.LBFGS, "max_iter": 20, "tol": 1e-6, "window": 3}
pbg.LBFGS.update(LBFGS_CFG)     # tighter stop than the pullback run: a geodesic, not a matched-budget replication
SIMPSON = (np.array([0.0, 0.5, 1.0]), np.array([1.0, 4.0, 1.0]) / 6.0)
FINE = (np.linspace(0, 1, 17), np.r_[0.5, np.ones(15), 0.5] / 16.0)


# ------------------------------------------------------------------ energies ---------------------------------------
def sqdist(A, B):
    """|a - b|^2 for A [.., M, k], B [N, k] -> [.., M, N] by the matmul expansion (no [M, N, k] temporary)."""
    return torch.clamp((A ** 2).sum(-1, keepdim=True) + (B ** 2).sum(-1) - 2 * A @ B.T, min=0.0)


class Energy:
    """E(z) for PCA-coordinate points z [n, M, k] of carrier i (c [n, N] = its fixed per-train-point offset)."""
    beta = BETA

    def g_sqrt(self, Z, c=None):
        E = self.E(Z, c)
        pt = torch.exp(torch.clamp(-(E - self.E_ref), max=50.0))
        return (((1 - self.beta) * pt + self.beta) ** -0.5)


class KNNEnergy(Energy):
    def __init__(self, Ztr, Rtr=None, k=KNN_K, dim_k=DIM_K):
        self.Ztr = torch.as_tensor(Ztr)
        self.Rtr = None if Rtr is None else np.asarray(Rtr, float)
        self.k = k
        Dfull = self._train_d2()
        np.fill_diagonal(Dfull, np.inf)
        srt = np.sqrt(np.sort(Dfull, axis=1)[:, :max(k, dim_k)])
        lr = np.log(srt[:, dim_k - 1:dim_k] / srt[:, :dim_k - 1])
        self.d_int = float(1.0 / np.mean(lr.mean(1)))            # MacKay-Ghahramani averaged Levina-Bickel
        loo = self.d_int * np.log(srt[:, :k].mean(1))
        self.E_ref = float(np.median(loo))
        self.loo_E = loo

    def _train_d2(self):
        Z = self.Ztr.numpy()
        d2 = ((Z[:, None] - Z[None]) ** 2).sum(-1)
        if self.Rtr is not None:
            r2 = (self.Rtr ** 2).sum(1)
            d2 = d2 + np.maximum(r2[:, None] + r2[None] - 2 * self.Rtr @ self.Rtr.T, 0)
        return d2

    def offsets(self, R):
        """c [n, N] = |r_i - r_j|^2 between carriers' residuals R [n, D] and the train residuals."""
        if self.Rtr is None:
            return None
        R = np.asarray(R, float)
        return torch.as_tensor(np.maximum((R ** 2).sum(1)[:, None] + (self.Rtr ** 2).sum(1)[None]
                                          - 2 * R @ self.Rtr.T, 0.0))

    def E(self, Z, c=None):
        d2 = sqdist(Z, self.Ztr)                                         # [n, M, N]
        if c is not None:
            d2 = d2 + c[:, None, :]
        d = torch.sqrt(torch.clamp(d2, min=1e-12))
        return self.d_int * torch.log(torch.topk(d, self.k, dim=-1, largest=False).values.mean(-1))


class KDEEnergy(Energy):
    def __init__(self, Ztr, folds, m=10, grid=np.logspace(-1.3, 0.5, 28)):
        Ztr = np.asarray(Ztr, float)
        self.m = m
        self.sd = Ztr[:, :m].std(0)
        U = Ztr[:, :m] / self.sd
        cv = []
        for h in grid:
            ll = []
            for f in np.unique(folds):
                A, B = U[folds != f], U[folds == f]
                ll.append(self._logpdf(B, A, h))
            cv.append(float(np.mean(np.concatenate(ll))))
        self.h = float(grid[int(np.argmax(cv))])
        self.cv = {"grid": grid.tolist(), "mean_heldout_loglik": cv, "h": self.h,
                   "at_grid_edge": bool(np.argmax(cv) in (0, len(grid) - 1))}
        self.U = torch.as_tensor(U)
        D = ((U[:, None] - U[None]) ** 2).sum(-1) / (2 * self.h ** 2)
        np.fill_diagonal(D, np.inf)
        loo = -np.log(np.exp(-(D - D.min(1, keepdims=True))).sum(1)) + D.min(1)
        self.E_ref = float(np.median(loo))
        self.loo_E = loo

    def _logpdf(self, B, A, h):
        D = ((B[:, None] - A[None]) ** 2).sum(-1) / (2 * h ** 2)
        mn = D.min(1, keepdims=True)
        return (-mn[:, 0] + np.log(np.exp(-(D - mn)).sum(1)) - np.log(len(A))
                - 0.5 * A.shape[1] * np.log(2 * np.pi * h ** 2))

    def offsets(self, R):
        return None

    def E(self, Z, c=None):
        u = Z[..., :self.m] / torch.as_tensor(self.sd)
        D = sqdist(u, self.U) / (2 * self.h ** 2)
        return -torch.logsumexp(-D, dim=-1)


class FlatEnergy(Energy):
    """G = I (Eq. 5): parity control."""
    E_ref = 0.0

    def offsets(self, R):
        return None

    def g_sqrt(self, Z, c=None):
        return torch.ones(Z.shape[:-1])


# ------------------------------------------------------------------ Eq. 4 --------------------------------------------
def length(P, energy, c=None, quad=SIMPSON):
    """Discrete Eq. 4: sum over segments of |dP| * quadrature of sqrt(g) along the segment. P [n, M, k] -> [n]."""
    q, w = (torch.as_tensor(a) for a in quad)
    dP = P[:, 1:] - P[:, :-1]
    pts = P[:, :-1, None] + q[None, None, :, None] * dP[:, :, None]       # [n, M-1, Q, k]
    n, S, Q, k = pts.shape
    gs = energy.g_sqrt(pts.reshape(n, S * Q, k), c).reshape(n, S, Q)
    return (torch.sqrt((dP ** 2).sum(-1) + 1e-18) * (gs * w).sum(-1)).sum(-1)


def geodesic(P0, energy, c=None, steps=200, log=lambda *a: None):
    """Minimise discrete Eq. 4 over the interior nodes of P0 [n, M, k] (endpoints fixed). Returns (P, info)."""
    P0 = torch.as_tensor(np.asarray(P0, float))
    a, b = P0[:, :1], P0[:, -1:]
    V = P0[:, 1:-1].clone().requires_grad_(True)

    def closure_fn():
        L = length(torch.cat([a, V, b], 1), energy, c).sum()
        L.backward()
        return L, []

    info = pbg.optimise(V, closure_fn, steps=steps, log=log)
    info.pop("per_t", None)
    info.pop("eval_losses", None)
    return torch.cat([a, V.detach(), b], 1).numpy(), info


def resample(P, K):
    """Resample polylines P [n, M, k] to K points uniform in Euclidean arc length (a geodesic is a curve; its node
    spacing is arbitrary under Eq. 4)."""
    P = np.asarray(P, float)
    out = np.empty((len(P), K, P.shape[-1]))
    for i, p in enumerate(P):
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        t = np.linspace(0, s[-1], K)
        out[i] = np.stack([np.interp(t, s, p[:, j]) for j in range(p.shape[1])], 1) if s[-1] > 0 else p[:1]
    return out


def endpoint_matched(Zs, end, M):
    """The spline path Zs [n, K, k] resampled to M nodes and bent onto the chord's end state by the arc-fraction
    linear correction f (end - Zs[-1])."""
    R = resample(Zs, M)
    s = np.r_[[np.r_[0, np.cumsum(np.linalg.norm(np.diff(r, axis=0), axis=1))] for r in R]]
    f = s / np.maximum(s[:, -1:], 1e-12)
    return R + f[..., None] * (np.asarray(end)[:, None] - R[:, -1:])


def random_init(chord, spline, rng):
    """Chord + a random smooth bump (3 sine modes, random 64-d directions), max deviation = the spline's."""
    n, M, k = chord.shape
    s = np.linspace(0, 1, M)
    bump = sum(np.sin((j + 1) * np.pi * s)[None, :, None] * rng.standard_normal((n, 1, k)) / (j + 1)
               for j in range(3))
    dev_sp = np.linalg.norm(spline - chord, axis=-1).max(1)
    scale = dev_sp / np.maximum(np.linalg.norm(bump, axis=-1).max(1), 1e-12)
    return chord + bump * scale[:, None, None]


# ------------------------------------------------------------------ scoring ------------------------------------------
def closest(P, ref):
    """Mean and max closest-point distance (A.9 / causalab) of each path P [n, K, k] to ref [n, K', k]."""
    d = np.array([np.linalg.norm(p - pbg.project_to_polyline(p, r), axis=1) for p, r in zip(P, ref)])
    return d.mean(1), d.max(1)


def bend_stats(P, chord, spline, U):
    """Deviation from the chord (path minus the chord point at the same arc fraction, chord_coords 'projected'):
    fraction of its squared norm in the ring plane U [2, k], and its cosine with the spline's own deviation."""
    proj, _ = mf.chord_coords(P)
    dev = P - proj
    tot = (dev ** 2).sum((1, 2))
    ring = ((dev @ U.T) ** 2).sum((1, 2))
    sp_matched = resample(spline, P.shape[1])
    sproj, _ = mf.chord_coords(sp_matched)
    sdev = sp_matched - sproj
    cos = (dev * sdev).sum((1, 2)) / np.maximum(np.sqrt(tot * (sdev ** 2).sum((1, 2))), 1e-12)
    return np.where(tot > 1e-18, ring / np.maximum(tot, 1e-300), np.nan), np.where(tot > 1e-18, cos, np.nan)


def ring_plane(cent_C):
    C = np.asarray(cent_C, float) - np.mean(cent_C, 0)
    _, S, Vt = np.linalg.svd(C, full_matrices=False)
    return Vt[:2], float((S[:2] ** 2).sum() / (S ** 2).sum())


def lg(P, energy, c):
    with torch.no_grad():
        return length(torch.as_tensor(P), energy, c, FINE).numpy()


EVAL_KEYS = ("probe_radius_min", "probe_err_path", "probe_err_to_target", "nearest_real_R", "behaviour_energy",
             "excess_to_nearest_real", "energy_to_nearest_real", "delta_norm_path")
HIGHER = {"probe_radius_min": True, "nearest_real_R": True}


def score_arm(Zp, x, resid, src, tgt, m, ctx, energies, cs, chord, spline, U):
    ev = rp2.evaluate(rp2.compose(m["pca"], Zp, resid), x, src, tgt, m, ctx)
    out = {q: ev[q] for q in EVAL_KEYS}
    L = lambda P: np.linalg.norm(np.diff(P, axis=1), axis=-1).sum(1)  # noqa: E731
    out["euclid_len_ratio_chord"] = L(Zp) / L(chord)
    for name, e in energies.items():
        out[f"LG_{name}"] = lg(Zp, e, cs[name])
    out["dist_to_spline_mean"], out["dist_to_spline_max"] = closest(Zp, spline)
    out["dist_to_chord_mean"], out["dist_to_chord_max"] = closest(Zp, chord)
    out["ring_plane_frac_of_bend"], out["bend_cos_with_spline"] = bend_stats(Zp, chord, spline, U)
    out["_wp_radius"] = ev["_wp_radius"]
    return out


# ------------------------------------------------------------------ one arc ------------------------------------------
def run_arc(d, layer, seed, n_clips, n_restart_clips, n_restarts, log, keep_paths=False):
    m = rp2.build(d, 64, "unsupervised", "contiguous", seed, n_controls=0, spline="smooth", behaviour_mode="spline",
                  subspace="pca", extend="cubic", aim="coord")
    probe_rows = d["role"] == "probe"
    probe = mf.ProbeReadout(d["X"][probe_rows], d["y"][probe_rows], True)
    test = np.flatnonzero(d["role"] == "test")
    ctx = {"periodic": True, "probe": probe, "near": mf.NearestRealReadout(d["X"][test], d["y"][test]),
           "near_ctx": None}
    ctx["floor"] = rp2.behaviour_floor(m["behaviour"], d["X"][test], d["y"][test])
    pca = m["pca"]
    Xtr = d["X"][m["knot"]].astype(float)
    Ztr, Rtr = pca.project(Xtr), pca.complement(Xtr)
    t0 = time.time()
    energies = {"knn": KNNEnergy(Ztr, Rtr), "kde": KDEEnergy(Ztr, d["fold"][m["knot"]])}
    flat = FlatEnergy()
    U, ring_var = ring_plane(m["cent"]["C"])
    top_m = energies["kde"].m
    calib = {"train_clips": int(m["knot"].sum()), "beta": BETA, "alpha": 1 - BETA,
             "knn": {"k": KNN_K, "d_int_levina_bickel_k10": energies["knn"].d_int, "E_ref": energies["knn"].E_ref},
             "kde": {"m_top_pcs": top_m, "bandwidth_cv": energies["kde"].cv, "E_ref": energies["kde"].E_ref},
             "ring_plane_centroid_var_frac": ring_var,
             "ring_plane_in_kde_pcs_frac": float((U[:, :top_m] ** 2).sum() / 2)}
    log(f"  arc s{seed}: energies built {time.time() - t0:.1f}s d_int={energies['knn'].d_int:.2f} "
        f"h={energies['kde'].h:.3f}")
    picks = rp2.pick_clips(d, m["held"], n_clips, seed)
    arms = ("chord", "spline", "spline_matched", "geo_knn_from_chord", "geo_knn_from_spline", "geo_kde_from_chord",
            "geo_kde_from_spline", "flat_from_spline")
    acc = {a: {} for a in arms}
    rst = {e: {} for e in energies}
    opt_info = {a: [] for a in arms if a.startswith(("geo", "flat"))}
    calib_test = {e: [] for e in energies}
    ids, groups = [], []
    example = None
    rng = np.random.default_rng(seed)
    for tgt, pick in picks.items():
        x, src = d["X"][pick].astype(float), d["y"][pick]
        Z, resid = pca.project(x), pca.complement(x)
        Zarms = rp2.subspace_arms(Z, src, tgt, m, 50)
        chord, spline = Zarms["linear_raw"], Zarms["manifold"]
        end = chord[:, -1]
        cs = {e: energies[e].offsets(resid) for e in energies}
        for e in energies:
            with torch.no_grad():
                Et = energies[e].E(torch.as_tensor(Z[:, None]), cs[e])[:, 0].numpy()
            calib_test[e].append(Et - energies[e].E_ref)
        M = N_FREE + 2
        init = {"chord": resample(chord, M), "spline": endpoint_matched(spline, end, M)}
        paths = {"chord": chord, "spline": spline, "spline_matched": resample(init["spline"], 50)}
        for e in energies:
            for i0 in ("chord", "spline"):
                P, info = geodesic(init[i0], energies[e], cs[e])
                paths[f"geo_{e}_from_{i0}"] = resample(P, 50)
                opt_info[f"geo_{e}_from_{i0}"].append({k: info[k] for k in ("outer_steps", "n_evals", "stop")})
        P, info = geodesic(init["spline"], flat)
        paths["flat_from_spline"] = resample(P, 50)
        opt_info["flat_from_spline"].append({k: info[k] for k in ("outer_steps", "n_evals", "stop")})
        for a in arms:
            sc = score_arm(paths[a], x, resid, src, tgt, m, ctx, energies, cs, chord, spline, U)
            for q, v in sc.items():
                acc[a].setdefault(q, []).append(v)
        for e in energies:
            acc[f"geo_{e}_from_spline"].setdefault("dist_to_geo_from_chord_mean", []).append(
                closest(paths[f"geo_{e}_from_spline"], paths[f"geo_{e}_from_chord"])[0])
        # random-restart null on the first n_restart_clips carriers
        sub = slice(0, n_restart_clips)
        for e in energies:
            csub = None if cs[e] is None else cs[e][sub]
            for r in range(n_restarts):
                P, _ = geodesic(random_init(init["chord"][sub], init["spline"][sub], rng), energies[e], csub)
                Pr = resample(P, 50)
                sc = score_arm(Pr, x[sub], resid[sub], src[sub], tgt, m, ctx, {e: energies[e]}, {e: csub},
                               chord[sub], spline[sub], U)
                ref = {i0: paths[f"geo_{e}_from_{i0}"][sub] for i0 in ("chord", "spline")}
                sc["dist_to_geo_from_chord_mean"] = closest(Pr, ref["chord"])[0]
                sc["dist_to_geo_from_spline_mean"] = closest(Pr, ref["spline"])[0]
                sc["LG_minus_geo_from_chord"] = sc[f"LG_{e}"] - acc[f"geo_{e}_from_chord"][f"LG_{e}"][-1][sub]
                for q, v in sc.items():
                    rst[e].setdefault(q, []).append(v)
        ids.append(d["df"]["id"].to_numpy()[pick])
        if example is None or keep_paths:
            j = 0
            example = example or {"target": float(tgt), "source": float(src[j]), "Z": Z[j], "resid": resid[j],
                                  "paths": {a: paths[a][j] for a in arms}, "U": U,
                                  "wp_radius": {a: acc[a]["_wp_radius"][-1][j] for a in arms}}
        log(f"  arc s{seed} target {tgt}: done {time.time() - t0:.0f}s")
    cat = {a: {q: np.concatenate(v) for q, v in qs.items()} for a, qs in acc.items()}
    ids = np.concatenate(ids)
    # A.9-style: do the two initialisations land on the same path?
    multi = {}
    out = {"seed": seed, "held_out_values": m["held"].tolist(), "angle_plane": m["plane"],
           "angle_source": m["curve"].coord_source, "calibration": calib,
           "calibration_carriers_E_minus_Eref": {e: {"median": float(np.median(np.concatenate(v))),
                                                     "p95": float(np.percentile(np.concatenate(v), 95))}
                                                 for e, v in calib_test.items()},
           "n_paths": int(len(ids)), "optimiser": {a: {"outer_steps_mean": float(np.mean([i["outer_steps"] for i in v])),
                                                     "stops": sorted({i["stop"] for i in v})}
                                                 for a, v in opt_info.items()},
           "arms": {a: {q: float(np.nanmean(v)) for q, v in qs.items() if not q.startswith("_")}
                    for a, qs in cat.items()},
           "arms_sd": {a: {q: float(np.nanstd(v)) for q, v in qs.items() if not q.startswith("_")}
                       for a, qs in cat.items()},
           "waypoint_radius_mean": {a: cat[a]["_wp_radius"].mean(0).tolist() for a in arms}}
    gaps = {}
    for a in [x for x in arms if x.startswith("geo")]:
        for ref in ("chord", "spline"):
            gaps[f"{a}_minus_{ref}"] = {q: rp2.paired_bootstrap(cat[a][q] - cat[ref][q], ids, seed=seed)
                                        for q in EVAL_KEYS + ("LG_knn", "LG_kde")}
    out["paired_gaps"] = gaps
    out["paired_gaps_note"] = ("arm minus reference on the same carriers and targets; 95% CI by paired bootstrap "
                               "over clips (a clip's targets resampled together); endpoint metrics "
                               "(probe_err_to_target, nearest_real_R) are identical for chord and geodesics by "
                               "construction (same end state)")
    for e in energies:
        a, b = f"geo_{e}_from_chord", f"geo_{e}_from_spline"
        multi[e] = {"LG_from_spline_minus_from_chord": rp2.paired_bootstrap(cat[b]["LG_" + e] - cat[a]["LG_" + e],
                                                                            ids, seed=seed),
                    "frac_paths_LG_differs_gt_1pct": float(np.mean(np.abs(cat[b]["LG_" + e] - cat[a]["LG_" + e])
                                                                   / cat[a]["LG_" + e] > 0.01)),
                    "dist_between_the_two_geodesics_mean": out["arms"][b]["dist_to_geo_from_chord_mean"],
                    "dist_to_spline_from_chord_vs_from_spline": [out["arms"][a]["dist_to_spline_mean"],
                                                                 out["arms"][b]["dist_to_spline_mean"]]}
    out["multimodality"] = multi
    out["restart_null"] = {e: {"n_clips_per_target": n_restart_clips, "n_restarts": n_restarts,
                               **{q: float(np.nanmean(np.concatenate(v))) for q, v in r.items()
                                  if not q.startswith("_")},
                               "frac_restarts_LG_below_geo_from_chord_by_1pct": float(np.mean(
                                   np.concatenate(r["LG_minus_geo_from_chord"]) <
                                   -0.01 * np.concatenate(r[f"LG_{e}"])))}
                           for e, r in rst.items()}
    return out, example, energies, m


def plot(example, energies, headline, path, layer):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    U, Z0 = example["U"], example["Z"]
    P = example["paths"]
    style = {"chord": ("#555555", "--"), "spline": ("#1f77b4", "-"), "geo_knn_from_chord": ("#d62728", "-"),
             "geo_knn_from_spline": ("#ff9896", ":"), "geo_kde_from_chord": ("#2ca02c", "-"),
             "geo_kde_from_spline": ("#98df8a", ":")}
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    allp = np.concatenate([(P[a] - Z0) @ U.T for a in style])
    lo, hi = allp.min(0), allp.max(0)
    pad = 0.35 * (hi - lo).max()
    ga = np.linspace(lo[0] - pad, hi[0] + pad, 70)
    gb = np.linspace(lo[1] - pad, hi[1] + pad, 70)
    A, B = np.meshgrid(ga, gb)
    grid = Z0 + A.reshape(-1, 1) * U[0] + B.reshape(-1, 1) * U[1]
    for i, e in enumerate(("knn", "kde")):
        en = energies[e]
        c = en.offsets(example["resid"][None])
        with torch.no_grad():
            E = en.E(torch.as_tensor(grid[None]), c)[0].numpy() - en.E_ref
        ax[i].contourf(A, B, E.reshape(A.shape), levels=14, cmap="Greys", alpha=0.6)
        for a, (col, ls) in style.items():
            q = (P[a] - Z0) @ U.T
            ax[i].plot(q[:, 0], q[:, 1], color=col, ls=ls, lw=2 if a.startswith(("chord", "spline")) or
                       e in a else 1, label=a, alpha=1 if (e in a or not a.startswith("geo")) else 0.35)
        ax[i].scatter([0], [0], c="k", s=20, zorder=5)
        ax[i].set_title(f"{e} energy E-E_ref (grey), ring plane through the carrier\n"
                        f"src {example['source']:.1f} -> tgt {example['target']:.1f} deg")
        ax[i].set_xlabel("ring axis 1 (PCA-64 units)")
        ax[i].set_ylabel("ring axis 2")
    ax[0].legend(fontsize=7, loc="best")
    s = np.linspace(0, 1, 50)
    for a, (col, ls) in style.items():
        ax[2].plot(s, headline["waypoint_radius_mean"][a], color=col, ls=ls, label=a)
    ax[2].set_xlabel("fraction of path (50 waypoints)")
    ax[2].set_ylabel("probe readout radius, mean over carriers")
    ax[2].set_title(f"point {layer}: readout radius along the path")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_arcs(s):
    if not s:
        return []
    lo, _, hi = s.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--n-clips", type=int, default=48)
    p.add_argument("--arcs", default="", help="extra contiguous-arc seeds, e.g. 1-16 (headline = seed 0)")
    p.add_argument("--arc-clips", type=int, default=16)
    p.add_argument("--restart-clips", type=int, default=8)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--figures-dir", default=str(ROOT / "figures"))
    a = p.parse_args(argv)
    log = lambda *s: print(*s, flush=True)  # noqa: E731
    d = load_inputs("direction", a.layer, "direction")
    head, ex, energies, m = run_arc(d, a.layer, 0, a.n_clips, a.restart_clips, a.restarts, log)
    res = Path(a.results_dir)
    res.mkdir(parents=True, exist_ok=True)
    out_path = res / f"p2_geodesic_direction_L{a.layer}.json"
    out = {"layer": a.layer, "dataset": "direction", "variable": "direction",
           "provenance": provenance(None, seeds={"seed": 0}, layer=a.layer, pool="meanpool", holdout="contiguous",
                                    spline="smooth", k=64, subspace="pca", K=50),
           "matches": f"results/p2_steer_direction_direction_L{a.layer}_contiguous_rawchord.json (seed 0)",
           "metric": "Goodfire 2605.05115 Eq. 4 length under Eq. 6 G_E = (alpha e^-E + beta)^-1 I",
           "arms": {"chord": "linear_raw arm (raw-centroid chord), K=50",
                    "spline": "manifold arm (smoothing spline, its OWN end state; differs from the chord's by the "
                              "sagitta)",
                    "spline_matched": "spline arm bent onto the chord's end state (the spline initialiser)",
                    "geo_{knn,kde}_from_{chord,spline}": "Eq. 4 minimiser under G_E, 50 free nodes, fixed chord "
                                                         "endpoints, resampled to 50 waypoints uniform in arc length",
                    "flat_from_spline": "same optimiser with G = I from the spline initialiser (parity: must be the "
                                        "chord)"},
           "optimiser": {"lbfgs": LBFGS_CFG, "n_free_nodes": N_FREE, "quadrature": "Simpson per segment (optimised); "
                         "16-subinterval trapezoid (reported LG_*)"},
           "headline": head}
    out_path.write_text(json.dumps(out, indent=1))
    Path(a.figures_dir).mkdir(parents=True, exist_ok=True)
    plot(ex, energies, head, Path(a.figures_dir) / f"fig_geodesic_direction_L{a.layer}.png", a.layer)
    arcs = []
    for s in parse_arcs(a.arcs):
        o, _, _, _ = run_arc(d, a.layer, s, a.arc_clips, 0, 0, log)
        o.pop("waypoint_radius_mean", None)
        o.pop("arms_sd", None)
        o.pop("restart_null", None)
        arcs.append(o)
        out["arcs"] = arcs
        out["arcs_note"] = (f"contiguous-arc seeds {a.arcs} (the results/arcs_rawchord blocks), "
                            f"{a.arc_clips} carriers per target, no restarts")
        out_path.write_text(json.dumps(out, indent=1))
    log("wrote", out_path)


if __name__ == "__main__":
    main()

"""Is the direction ring a circle, an ellipse or a bent line, and does its local curvature predict where the spline
beats the chord? Direction set, train rows only, stored meanpool, PCA-64 and per-value centroids as in
scripts/run_geometry_checks.py.

Per point (2, 4, ..., 24; figures at 8 / 12 / 22):
  1. centroid PCA plane: Fitzgibbon conic, polar r(phi) fit, supervised chart c + A[cos, sin]; semi-axes, b/a,
     eccentricity, orientation; RMS residual of ellipse vs circle vs count-weighted periodic smoothing spline, in
     units of the within-value centroid noise;
  2. full 1024-D: centroid-covariance eigenvalues (q = sqrt(l2/l1), raw and minus centroid noise), 2-theta Fourier
     power of r(theta), variance outside the top-2 plane, harmonics of the 3rd/4th centroid axes;
  3. angle distortion of atan2(PC2, PC1) (centroid plane, and the pipeline's plane as stored in p2_geometry_*);
  4. at points 12 and 22: ellipse / spline curvature at the midpoint of each held-out 45-degree arc of results/arcs
     vs the spline-minus-chord probe min-radius gap (Spearman);
  5. the same shape numbers for random-init ViT-L and for planted controls (circle and b/a = 0.5 ellipse planted into
     real point-12 train activations with the geometry_checks.planted_ring_control recipe; dims.planted_ring clean
     and sheared x30).
Writes results/p2_ellipse_direction.json, figures/fig4g_ellipse_direction.png, figures/fig4g_ellipse_arc_curvature.png.

  PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/run_ellipse.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from wm import ellipse as el
from wm import geometry_checks as gc
from wm import manifold as mf
from wm.data import PROJECT_ROOT
from wm.dims import planted_ring
from wm.p2_data import load_inputs
from wm.provenance import provenance

POINTS = tuple(range(2, 25, 2))
FIG_POINTS = (8, 12, 22)
ARC_POINTS = (12, 22)
RANDOM_DIR = PROJECT_ROOT / "artifacts" / "activations" / "direction" / "random"


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))


def analyse(X, y, k=64, keep_plot=False):
    """All shape numbers for one activation matrix X [n, D] (train rows) with direction labels y (degrees)."""
    X, y = np.asarray(X, np.float64), np.asarray(y, float)
    pca = mf.fit_pca(X, k)
    Z = pca.project(X)
    cent = mf.centroids(Z, y)
    values, count, C = cent["values"], cent["count"], cent["C"]
    th = np.radians(values)
    idx = np.searchsorted(values, y)

    # ---- 1. shape fits in two planes of the PCA-64 centroids: the ring plane (span of the k = 1 chart A, i.e. the
    # plane of the cos / sin component) and the variance top-2 plane (the geometry_checks "centroid plane")
    mu = C.mean(0)
    A64 = el.chart_fit(C, th)["A"]
    ring_basis = np.linalg.svd(A64, full_matrices=False)[0][:, :2]           # [64, 2], major axis first
    var_basis = np.linalg.svd(C - mu, full_matrices=False)[2][:2].T
    Rz = Z - C[idx]
    ring = plane_fits((C - mu) @ ring_basis, Rz @ ring_basis, th, count, len(Z))
    var2 = plane_fits((C - mu) @ var_basis, Rz @ var_basis, th, count, len(Z))
    angle_between = np.degrees(np.arccos(np.clip(np.linalg.svd(ring_basis.T @ var_basis, compute_uv=False), -1, 1)))

    # ---- 2. full space
    Xc = np.array([X[idx == i].mean(0) for i in range(len(values))])
    Rfull = X - Xc[idx]
    Sw = Rfull.T @ Rfull / (len(X) - len(values))
    ncov = Sw * np.mean(1 / count)
    spec = el.centroid_spectrum(Xc, noise_cov=ncov)
    four = el.radius_fourier(Xc, th, noise_r2=np.trace(Sw) / count)
    harmv = el.harmonic_variance(Xc, th, noise_trace=np.trace(ncov))
    chq, _ = el.chart_q_full(Xc, th, noise_trace=np.trace(ncov))
    U, S, _ = np.linalg.svd(Xc - Xc.mean(0), full_matrices=False)
    harm = {}
    for j in (0, 1, 2, 3):
        f = np.abs(np.exp(-1j * np.outer(np.arange(1, 9), th)) @ (U[:, j] * S[j])) ** 2
        harm[f"axis{j + 1}"] = {"dominant_k": int(np.argmax(f) + 1), "share_k1": float(f[0] / f.sum()),
                                "share_k2": float(f[1] / f.sum()), "share_k3": float(f[2] / f.sum())}
    full = {"top2_eigen": {**spec, "note": "variance top-2 plane of the 1024-D centroids; when the k = 2 (saddle) "
                                           "axis outranks the ring's minor axis, sqrt(l2/l1) is NOT an axis ratio"},
            "k1_chart_axis_ratio": chq, "harmonic_variance": harmv, "radius_fourier": four,
            "centroid_axis_harmonics": harm,
            "k2_axis_rank": int(next((j + 1 for j in range(4) if harm[f"axis{j + 1}"]["dominant_k"] == 2), 0))}

    # ---- 3. angle distortion
    choice = mf.choose_angle_source(C, values)
    pipe_plane = choice["plane"]
    angle = {"pipeline_plane": {"plane": pipe_plane, "pipeline_angle_source": choice["angle"],
                                **el.angle_distortion(mf.unsupervised_angle(C, pipe_plane), th)},
             "stored_checks_recomputed": {pl: {"mean_dev_deg": choice["checks"][pl]["mean_dev_deg"],
                                               "circular_corr": choice["checks"][pl]["circular_corr"]}
                                          for pl in ("activation", "centroid")}}
    if choice["angle"] == "labels":
        angle["flag"] = ("pipeline fell back to the labels angle at this point (atan2 in both PCA planes fails the "
                         "circular-corr / order check); its atan2 fit is reported but is not a ring angle")

    s_ring = ring["summary"]
    out = {"n_train": int(len(X)), "n_values": int(len(values)), "min_count": int(count.min()),
           "ring_plane": ring, "variance_top2_plane": var2,
           "principal_angles_ring_vs_top2_plane_deg": np.sort(angle_between).tolist(),
           "full_space": full, "angle": angle,
           "summary": {"q_conic": s_ring["q_conic"], "q_polar": s_ring["q_polar"], "q_chart_plane": s_ring["q_chart"],
                       "q_k1_chart_full": chq["q"], "q_k1_chart_full_noise_corrected": chq["q_noise_corrected"],
                       "q_eig_top2_full": spec["q_eig"], "q_eig_top2_full_noise_corrected": spec["q_eig_noise_corrected"],
                       "q_r2_fourier_noise_corrected": four["q_from_r2_noise_corrected"],
                       "q_angle_exact": s_ring["q_angle_exact"], "q_angle_linear": s_ring["q_angle_linear"],
                       "r2_2theta_angle": s_ring["r2_2theta_angle"],
                       "q_angle_pipeline_plane": angle["pipeline_plane"]["q_exact"],
                       "r2_2theta_angle_pipeline_plane": angle["pipeline_plane"]["r2_2theta"],
                       "share_2theta_of_radius_power": four["share_2theta_of_k_ge_1"],
                       "p2_over_p0": four["p2_over_p0"],
                       "var_share_k1": harmv["share_noise_corrected"][0], "var_share_k2": harmv["share_noise_corrected"][1],
                       "bend_ratio_k2_over_k1": harmv["bend_ratio_k2_over_k1_noise_corrected"],
                       "k2_axis_rank": full["k2_axis_rank"],
                       "frac_outside_top2_noise_corrected": spec["frac_outside_top2_noise_corrected"],
                       "q_conic_variance_top2_plane": var2["summary"]["q_conic"],
                       "q_angle_variance_top2_plane": var2["summary"]["q_angle_exact"],
                       "resid_geo_over_noise": s_ring["resid_geo_over_noise"]}}
    if keep_plot:
        out["_plot"] = ring.pop("_plot")
    else:
        ring.pop("_plot")
    var2.pop("_plot")
    return out, pca, cent, choice


def plane_fits(P, Rw, th, count, n):
    """Conic, polar, chart, circle and smoothing-spline fits of 2-D centroids P [m, 2] at true angles th, with the
    within-value residuals Rw [n, 2] in the same plane (noise unit), plus the atan2 distortion in this plane."""
    m = len(P)
    sd = np.sqrt((Rw ** 2).sum(0) / (n - m))                                 # pooled within-value SD per axis
    unit = rms(np.sqrt((sd ** 2).mean() / count))                            # per-axis centroid SE
    con = el.conic_fit(P)
    pol = el.polar_fit(P, con)
    cir = el.circle_fit(P)
    ch = el.chart_fit(P, th)
    cc_pred = el.circle_chart_fit(P, th)
    sp = mf.SmoothSpline(th, P, count, sd, True)
    sp_dense = sp(np.linspace(th[0], th[0] + 2 * np.pi, 4000, endpoint=False))
    geo = {"ellipse_conic": el.curve_distance(P, el.ellipse_points(con["center"], con["a"], con["b"], con["angle"])),
           "ellipse_polar": el.curve_distance(P, el.ellipse_points(pol["center"], pol["a"], pol["b"], pol["angle"])),
           "circle": np.abs(cir["resid"]), "spline_smooth": el.curve_distance(P, sp_dense)}
    vec = {"circle_chart": np.linalg.norm(P - cc_pred, axis=1), "ellipse_chart": np.linalg.norm(P - ch["pred"], axis=1),
           "spline_smooth": np.linalg.norm(P - sp(th), axis=1)}
    ang = el.angle_distortion(np.arctan2(P[:, 1] - P[:, 1].mean(), P[:, 0] - P[:, 0].mean()), th)
    geo_rms = {k: rms(v) / unit for k, v in geo.items()}
    return {"noise_unit_per_axis_centroid_se": unit,
            "conic": {q: con[q] for q in ("a", "b", "q", "ecc", "angle", "center")},
            "polar_fit": {q: pol[q] for q in ("a", "b", "q", "ecc", "angle", "center", "rms_radial")},
            "chart": {"a": ch["a"], "b": ch["b"], "q": ch["q"], "angle": ch["angle"]},
            "circle": {"R": cir["R"], "center": cir["center"]},
            "geometric_rms_over_noise": geo_rms,
            "vector_rms_at_true_angle_over_noise": {k: rms(v) / unit for k, v in vec.items()},
            "angle_distortion": ang,
            "params": {"circle": 3, "ellipse": 5, "circle_chart": 4, "ellipse_chart": 6,
                       "spline_smooth": "count-weighted, s = 64 (chi-square = number of knots by construction)"},
            "residual_note": ("geometric = distance to the fitted closed curve (a correct model with pure centroid "
                              "noise gives ~1); vector = distance to the model point at the true angle (~sqrt(2)); "
                              "the smoothing spline is tuned to chi-square = m, so ~1 / ~sqrt(2) by construction"),
            "summary": {"q_conic": con["q"], "q_polar": pol["q"], "q_chart": ch["q"], "q_angle_exact": ang["q_exact"],
                        "q_angle_linear": ang["q_linear"], "r2_2theta_angle": ang["r2_2theta"],
                        "resid_geo_over_noise": {k: geo_rms[k] for k in ("ellipse_conic", "circle", "spline_smooth")}},
            "_plot": {"P": P, "values": np.degrees(th), "con": con, "cir": cir, "sp_dense": sp_dense}}


# ---------------------------------------------------------------- 4. arcs -------------------------------------------

def arc_table(point, pca_cent_choice, results_dir):
    pca, cent, choice = pca_cent_choice
    C, values = cent["C"], cent["values"]
    ch64 = el.chart_fit(C, np.radians(values))
    ab = np.sqrt(ch64["a"] * ch64["b"])
    curve = mf.fit_curve(cent, True, angle=choice["angle"], plane=choice["plane"], spline="smooth")
    thv = np.radians(values)
    H = np.linalg.lstsq(np.stack([np.ones_like(thv), np.cos(thv), np.sin(thv), np.cos(2 * thv), np.sin(2 * thv)], 1),
                        C, rcond=None)[0]                                    # k <= 2 harmonic curve (ring + saddle)

    def kappa_k12(t):
        d1 = -np.sin(t) * H[1] + np.cos(t) * H[2] - 2 * np.sin(2 * t) * H[3] + 2 * np.cos(2 * t) * H[4]
        d2 = -np.cos(t) * H[1] - np.sin(t) * H[2] - 4 * np.cos(2 * t) * H[3] - 4 * np.sin(2 * t) * H[4]
        return float(el.curve_curvature(d1[None], d2[None])[0])

    arcs = {}
    for f in sorted(Path(results_dir, "arcs").glob(f"L{point}_s*/p2_steer_direction_direction_L{point}_contiguous.json")):
        d = json.loads(f.read_text())
        h = d["holdout"]
        gap = d["gaps"]["manifold_minus_linear"]["probe_radius_min"]["mean"]
        sag = float(np.mean([s["sagitta"] for s in d["sagitta_per_target"]]))
        arcs.setdefault(h["block_start_index"], []).append({"seed": h["seed"], "first": h["block_first_value"],
                                                             "gap": gap, "sagitta": sag})
    rows = []
    for start, runs in sorted(arcs.items()):
        mid = (runs[0]["first"] + 3.5 * 360.0 / 64) % 360.0
        t = curve.coord_of_value(mid)
        d1, d2 = curve.spline(np.atleast_1d(t), 1), curve.spline(np.atleast_1d(t), 2)
        rows.append({"block_start_index": int(start), "seeds": [r["seed"] for r in runs], "mid_deg": float(mid),
                     "kappa_ellipse_chart64": float(el.chart_curvature(ch64["A"], np.radians(mid))[0]),
                     "kappa_ellipse_norm": float(el.chart_curvature(ch64["A"], np.radians(mid))[0] * ab),
                     "kappa_spline_smooth": float(el.curve_curvature(d1, d2)[0]),
                     "kappa_k12_harmonic": kappa_k12(np.radians(mid)),
                     "mean_stored_sagitta": float(np.mean([r["sagitta"] for r in runs])),
                     "gap_probe_radius_min": float(np.mean([r["gap"] for r in runs]))})
    gap = [r["gap_probe_radius_min"] for r in rows]
    corr = {}
    for key in ("kappa_ellipse_chart64", "kappa_k12_harmonic", "kappa_spline_smooth", "mean_stored_sagitta"):
        rho, p = spearmanr([r[key] for r in rows], gap)
        corr[key] = {"spearman_rho": float(rho), "p": float(p), "n": len(rows)}
    return {"n_distinct_arcs": len(rows), "chart64_q": ch64["q"], "rows": rows, "spearman_vs_gap": corr,
            "gap_field": "gaps.manifold_minus_linear.probe_radius_min.mean (duplicate arcs averaged)"}


# ---------------------------------------------------------------- 5. controls ---------------------------------------

def planted_in_real(X, labels, q, r=16.0, seed=0, bend=0.0):
    """geometry_checks.planted_ring_control recipe (same Q, sd and stratified theta for this seed), radius r within-SD
    along Q[:, 0] and q * r along Q[:, 1]; bend > 0 adds a saddle bend * r * cos 2 theta along a third direction."""
    X = np.asarray(X, np.float64)
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((X.shape[1], 2)))[0]
    Q3 = np.linalg.qr(np.column_stack([Q, np.random.default_rng(seed + 1).standard_normal(X.shape[1])]))[0][:, 2]
    grid = np.arange(64) * 360.0 / 64
    order = np.argsort(np.asarray(labels, float), kind="stable")
    theta = np.empty(len(X))
    for i in range(0, len(X), 64):
        blk = order[i:i + 64]
        theta[blk] = rng.permutation(grid)[:len(blk)]
    sd = gc.within_value_sd(X, labels, Q)
    t = np.radians(theta)
    ring = np.stack([np.cos(t), q * np.sin(t)], 1) @ Q.T + bend * np.cos(2 * t)[:, None] * Q3[None]
    return X + r * sd * ring, theta, r * sd


def plot_rings(per_point):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, L in zip(axes, FIG_POINTS):
        pp = per_point[L]["_plot"]
        P = pp["P"]
        sc = ax.scatter(*P.T, c=pp["values"], cmap="twilight", s=16, zorder=3)
        con, cir = pp["con"], pp["cir"]
        E = el.ellipse_points(con["center"], con["a"], con["b"], con["angle"], 400)
        ax.plot(*np.vstack([E, E[:1]]).T, color="C3", lw=1.4, label=f"ellipse b/a={con['q']:.2f}")
        u = np.linspace(0, 2 * np.pi, 400)
        ax.plot(cir["center"][0] + cir["R"] * np.cos(u), cir["center"][1] + cir["R"] * np.sin(u), color="C0",
                lw=1.1, ls="--", label="circle")
        sd = np.vstack([pp["sp_dense"], pp["sp_dense"][:1]])
        ax.plot(*sd.T, color="0.4", lw=0.8, label="smoothing spline")
        s = per_point[L]["summary"]
        ax.set_title(f"point {L}: resid/noise ell {s['resid_geo_over_noise']['ellipse_conic']:.2f}, "
                     f"circ {s['resid_geo_over_noise']['circle']:.2f}, spl {s['resid_geo_over_noise']['spline_smooth']:.2f}",
                     fontsize=8)
        ax.set_xlabel("ring plane, major axis")
        ax.set_ylabel("ring plane, minor axis")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=7, loc="lower right")
    fig.colorbar(sc, ax=axes[:3], shrink=0.7, label="direction (deg)")
    ax = axes[3]
    Ls = sorted(per_point)
    for key, lab, st in (("q_conic", "conic fit, ring plane", "-o"), ("q_polar", "r(phi) fit, ring plane", "-s"),
                         ("q_k1_chart_full_noise_corrected", "k=1 chart singular values, 1024-D", "-^"),
                         ("q_angle_exact", "atan2 distortion, ring plane", "-d"),
                         ("q_eig_top2_full_noise_corrected", "sqrt(l2/l1) top-2, 1024-D (saddle-contaminated)", "--x")):
        ax.plot(Ls, [per_point[L]["summary"][key] for L in Ls], st, ms=4, label=lab)
    ax.plot(Ls, [per_point[L]["summary"]["var_share_k2"] for L in Ls], "-", color="C7", lw=2, alpha=0.5,
            label="variance share of k=2 (saddle bend)")
    return fig, ax


def main(args):
    res_dir, fig_dir = Path(args.results_dir), Path(args.figures_dir)
    per_point, keep, random_pp, stored = {}, {}, {}, {}
    for L in POINTS:
        d = load_inputs("direction", L)
        tr = d["is_train"]
        out, pca, cent, choice = analyse(d["X"][tr], d["y"][tr], keep_plot=L in FIG_POINTS)
        per_point[L] = out
        if L in ARC_POINTS:
            keep[L] = (pca, cent, choice)
        g = res_dir / f"p2_geometry_direction_L{L}.json"
        if g.exists():
            ang = json.loads(g.read_text())["angle"]
            stored[L] = {pl: {"stored_mean_dev_deg": ang[pl]["mean_dev_deg"],
                              "recomputed_mean_dev_deg": out["angle"]["stored_checks_recomputed"][pl]["mean_dev_deg"]}
                         for pl in ("activation", "centroid")}
            stored[L]["stored_source"] = ang["source"]
        dr = load_inputs("direction", L, act_dir=RANDOM_DIR)
        ro, *_ = analyse(dr["X"][dr["is_train"]], dr["y"][dr["is_train"]])
        random_pp[L] = ro
        s = out["summary"]
        print(f"L{L:2d} q conic {s['q_conic']:.3f} polar {s['q_polar']:.3f} k1chart {s['q_k1_chart_full_noise_corrected']:.3f} "
              f"top2eig {s['q_eig_top2_full_noise_corrected']:.3f} angle {s['q_angle_exact']:.3f} "
              f"(R2 {s['r2_2theta_angle']:.2f}) k2share {s['var_share_k2']:.3f} | random conic "
              f"{ro['summary']['q_conic']:.3f} k1chart {ro['summary']['q_k1_chart_full_noise_corrected']:.3f}", flush=True)

    arcs = {L: arc_table(L, keep[L], res_dir) for L in ARC_POINTS}

    # controls
    d12 = load_inputs("direction", 12)
    X12, y12 = d12["X"][d12["is_train"]], d12["y"][d12["is_train"]]
    controls = {}
    for name, q, bend in (("planted_circle_in_real_L12", 1.0, 0.0), ("planted_ellipse_q0.5_in_real_L12", 0.5, 0.0),
                          ("planted_saddle_circle_bend1.2_in_real_L12", 1.0, 1.2)):
        Xp, th, rad = planted_in_real(X12, y12, q, bend=bend)
        o, *_ = analyse(Xp, th)
        controls[name] = {"planted_q": q, "planted_bend_over_radius": bend, "planted_major_radius": float(rad), "recipe":
                          "geometry_checks.planted_ring_control (seed 0 Q, within-SD, stratified theta), r = 16",
                          "summary": o["summary"]}
    for name, kw in (("dims_clean_ring", {}), ("dims_sheared_ring_x30", {"shear": 30})):
        Xs, ths = planted_ring(**kw)
        o, *_ = analyse(Xs, ths)
        chart_full = gc.fit_circular_chart(Xs, ths)["A"]
        sv = np.linalg.svd(chart_full, compute_uv=False)
        controls[name] = {"planted": {k: str(v) for k, v in kw.items()}, "chart_full_space_q": float(sv[1] / sv[0]),
                          "summary": o["summary"]}

    out = {"dataset": "direction", "points": list(POINTS), "figure_points": list(FIG_POINTS),
           "provenance": provenance(None, seeds={"planted": 0, "dims_planted_ring": 0}, pool="meanpool",
                                    subspace="pca", k=64, rows="train (all 5 train folds)"),
           "estimators": el.__doc__.strip(),
           "per_point": {str(L): {k: v for k, v in o.items() if k != "_plot"} for L, o in per_point.items()},
           "random_init": {str(L): {"summary": o["summary"], "angle_source": o["angle"]["pipeline_plane"]["pipeline_angle_source"]}
                           for L, o in random_pp.items()},
           "stored_angle_check": {str(L): v for L, v in stored.items()},
           "arcs": {str(L): v for L, v in arcs.items()},
           "controls": controls}
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "p2_ellipse_direction.json").write_text(json.dumps(out, indent=1))

    fig, ax = plot_rings(per_point)
    Ls = sorted(random_pp)
    ax.plot(Ls, [random_pp[L]["summary"]["q_conic"] for L in Ls], ":o", color="0.5", ms=3, label="random-init, conic")
    ax.plot(Ls, [random_pp[L]["summary"]["q_k1_chart_full_noise_corrected"] for L in Ls], ":^", color="0.3", ms=3,
            label="random-init, k=1 chart")
    ax.axhline(1, color="k", lw=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("point (0 = embedding)")
    ax.set_ylabel("axis ratio b/a")
    ax.set_title("axis ratio vs point", fontsize=9)
    ax.legend(fontsize=6)
    fig.savefig(fig_dir / "fig4g_ellipse_direction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, L in zip(axes, ARC_POINTS):
        rows = arcs[L]["rows"]
        k = [r["kappa_ellipse_norm"] for r in rows]
        g = [r["gap_probe_radius_min"] for r in rows]
        ax.scatter(k, g, s=22)
        for r in rows:
            ax.annotate(f"{r['mid_deg']:.0f}", (r["kappa_ellipse_norm"], r["gap_probe_radius_min"]), fontsize=6)
        c = arcs[L]["spearman_vs_gap"]["kappa_ellipse_chart64"]
        ax.set_title(f"point {L}: Spearman rho {c['spearman_rho']:.2f} (p {c['p']:.2f}, n {c['n']})", fontsize=9)
        ax.set_xlabel("ellipse curvature at arc midpoint x sqrt(ab)")
        ax.set_ylabel("spline - chord probe min radius")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4g_ellipse_arc_curvature.png", dpi=150)
    plt.close(fig)

    for L in ARC_POINTS:
        print(f"arcs L{L}: " + ", ".join(f"{k} rho {v['spearman_rho']:.2f} p {v['p']:.3f}"
                                          for k, v in arcs[L]["spearman_vs_gap"].items()))
    for n, c in controls.items():
        s = c["summary"]
        print(f"{n}: conic {s['q_conic']:.3f} k1chart {s['q_k1_chart_full_noise_corrected']:.3f} top2eig "
              f"{s['q_eig_top2_full_noise_corrected']:.3f} angle {s['q_angle_exact']:.3f} k2share {s['var_share_k2']:.3f}")
    return out


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--figures-dir", default=str(PROJECT_ROOT / "figures"))
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse())

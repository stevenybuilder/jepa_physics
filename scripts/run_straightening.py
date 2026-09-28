"""Perceptual straightening in V-JEPA 2 (PRIORITIES S9; the open question of Musa et al., arXiv 2609.01551).

Curvature (Hénaff, Goris & Simoncelli 2019, Nat. Neurosci.): for a trajectory x_1..x_T, displacements v_t = x_{t+1} - x_t,
c = mean_t arccos(v_t . v_{t+1} / |v_t||v_{t+1}|), in degrees. T = 8 token time steps of the stored `timepool`
[N, 26, 8, 1024] (spatial mean per tubelet). Pixel reference: the cached 32x32 grayscale frames (artifacts/pixels),
frames (2t, 2t+1) averaged into the same 8 tubelets -> [N, 8, 1024]. Straightening index = pixel - latent curvature.
Nulls: random walks with each clip's step norms (isotropic in 1024-d; and covariance-matched: directions drawn from
the pooled displacement second moment of that point). Bootstrap CIs over clips (B = 2000).
-> results/p5_straightening.json, figures/fig_straightening.png
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ACT = ROOT / "artifacts" / "activations"
PIX = ROOT / "artifacts" / "pixels"
RES, FIG = ROOT / "results", ROOT / "figures"
ZONE = list(range(8, 13))
B = 2000
POINTS = 26


def curvature(X, eps=1e-8):
    """X [..., T, D] -> mean angle (degrees) between successive displacement vectors, shape [...].
    Steps with |v| < eps are dropped (nan-mean); all-degenerate -> nan."""
    V = np.diff(np.asarray(X, np.float64), axis=-2)
    n = np.linalg.norm(V, axis=-1)
    cos = (V[..., :-1, :] * V[..., 1:, :]).sum(-1) / (n[..., :-1] * n[..., 1:] + 1e-300)
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    ang = np.where((n[..., :-1] > eps) & (n[..., 1:] > eps), ang, np.nan)
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sw:
        sw.filter(RuntimeWarning)
        return np.nanmean(ang, axis=-1)


def null_curvature(X, rng, cov_basis=None):
    """Random walk with the step norms of each trajectory in X [N, T, D]. Directions isotropic, or drawn as
    z @ cov_basis (cov_basis = diag(S) Vt / sqrt(n) of the pooled displacements) when given."""
    V = np.diff(np.asarray(X, np.float64), axis=-2)
    n = np.linalg.norm(V, axis=-1, keepdims=True)
    if cov_basis is None:
        G = rng.standard_normal(V.shape)
    else:
        G = rng.standard_normal(V.shape[:-1] + (cov_basis.shape[0],)) @ cov_basis
    G = G / np.linalg.norm(G, axis=-1, keepdims=True) * n
    walk = np.concatenate([np.zeros(V.shape[:-2] + (1, V.shape[-1])), np.cumsum(G, axis=-2)], axis=-2)
    return curvature(walk)


def boot_mean(C, rng, b=B):
    """C [N, ...] -> dict(mean, lo, hi) over clips (nan-aware), percentile 95% CI."""
    C = np.asarray(C, np.float64)
    idx = rng.integers(0, len(C), (b, len(C)))
    bm = np.stack([np.nanmean(C[i], axis=0) for i in idx])
    return {"mean": np.nanmean(C, axis=0).tolist(), "lo": np.nanpercentile(bm, 2.5, axis=0).tolist(),
            "hi": np.nanpercentile(bm, 97.5, axis=0).tolist()}, bm


def boot_stat(fn, n, rng, b=B):
    vals = np.array([fn(rng.integers(0, n, n)) for _ in range(b)])
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def pixel_traj(dataset):
    gray = np.load(PIX / f"{dataset}.npy")                        # [N, 16, 32, 32] uint8
    x = gray.astype(np.float32).reshape(len(gray), 8, 2, -1).mean(2) / 255.0
    cen = np.load(PIX / f"{dataset}_centroid.npy").astype(np.float64)   # [N, 16, 2]
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sw:
        sw.filter(RuntimeWarning)
        c8 = np.nanmean(cen.reshape(len(cen), 8, 2, 2), axis=2)
    return x, c8


def latent_curv(path, rows):
    tp = np.load(path, mmap_mode="r")
    out = np.empty((len(rows), POINTS))
    for p in range(POINTS):
        out[:, p] = curvature(np.asarray(tp[rows, p], np.float32))
    return out


def null_curves(path, rows, rng):
    tp = np.load(path, mmap_mode="r")
    iso, cov = np.empty((len(rows), POINTS)), np.empty((len(rows), POINTS))
    for p in range(POINTS):
        X = np.asarray(tp[rows, p], np.float32).astype(np.float64)
        iso[:, p] = null_curvature(X, rng)
        D = np.diff(X, axis=1).reshape(-1, X.shape[-1])
        _, S, Vt = np.linalg.svd(D, full_matrices=False)
        cov[:, p] = null_curvature(X, rng, (S[:, None] * Vt) / np.sqrt(len(D)))
    return iso, cov


def zone_test(C, rng):
    """Is the depth minimum of the mean curve inside points 8-12, or does curvature keep falling to the output?"""
    m = np.nanmean(C, 0)
    idx = rng.integers(0, len(C), (B, len(C)))
    bm = np.stack([np.nanmean(C[i], 0) for i in idx])
    am = bm.argmin(1)
    late = list(range(13, POINTS))
    d_zone_late = bm[:, ZONE].min(1) - bm[:, late].min(1)
    d_zone_out = bm[:, ZONE].min(1) - bm[:, 25]
    rho_late = spearmanr(late, m[late])[0]
    return {"argmin_point": int(m.argmin()), "min_curvature_deg": float(m.min()),
            "boot_argmin_counts": {str(k): int(v) for k, v in zip(*np.unique(am, return_counts=True))},
            "p_argmin_in_zone": float(np.isin(am, ZONE).mean()),
            "zone_min_minus_late_min_deg": {"mean": float(m[ZONE].min() - m[late].min()),
                                            "ci95": np.percentile(d_zone_late, [2.5, 97.5]).tolist()},
            "zone_min_minus_point25_deg": {"mean": float(m[ZONE].min() - m[25]),
                                           "ci95": np.percentile(d_zone_out, [2.5, 97.5]).tolist()},
            "spearman_depth_vs_curvature_points13_25": float(rho_late)}


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def auroc(pos, neg):
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    from wm.data import load_table
    dfd, dfa = load_table("direction"), load_table("acceleration")
    for sub, df in [("direction/vjepa2", dfd), ("direction/vjepa2_timerev", dfd), ("direction/random", dfd),
                    ("acceleration/vjepa2", dfa)]:
        assert json.loads((ACT / sub / "ids.json").read_text()) == df["id"].astype(int).tolist(), sub
    vel = np.flatnonzero(dfd["motion"].to_numpy() == "velocity")
    acc_d = np.flatnonzero(dfd["motion"].to_numpy() == "acceleration")
    all_a = np.arange(len(dfa))
    speed = dfd["speed_mps"].to_numpy(float)[vel]

    pxd, cend = pixel_traj("direction")
    pxa, cena = pixel_traj("acceleration")
    pix = {"constvel": curvature(pxd[vel]), "accel_direction_set": curvature(pxd[acc_d]),
           "accel_acceleration_set": curvature(pxa)}
    cart = {"constvel": curvature(cend[vel]), "accel_direction_set": curvature(cend[acc_d]),
            "accel_acceleration_set": curvature(cena)}

    F, R, RI = ACT / "direction/vjepa2/timepool.npy", ACT / "direction/vjepa2_timerev/timepool.npy", \
        ACT / "direction/random/timepool.npy"
    lat = {"constvel": latent_curv(F, vel), "accel_direction_set": latent_curv(F, acc_d),
           "accel_acceleration_set": latent_curv(ACT / "acceleration/vjepa2/timepool.npy", all_a),
           "reversed_constvel": latent_curv(R, vel), "reversed_accel_direction_set": latent_curv(R, acc_d),
           "random_init_constvel": latent_curv(RI, vel), "random_init_accel_direction_set": latent_curv(RI, acc_d)}
    print(f"curvatures done {time.time() - t0:.0f}s", flush=True)
    iso, cov = null_curves(F, vel, rng)
    lat["null_isotropic_constvel"], lat["null_covmatched_constvel"] = iso, cov
    print(f"nulls done {time.time() - t0:.0f}s", flush=True)

    out = {"definition": "Henaff, Goris & Simoncelli 2019 (Nat. Neurosci. 22:984): curvature = mean over t of the "
                         "angle (deg) between successive displacement vectors v_t = x_{t+1} - x_t; 8 time steps -> "
                         "7 displacements -> 6 angles per trajectory",
           "question": "Musa et al., arXiv 2609.01551: straightening / oscillation of trajectories with depth, "
                       "no definitive explanation",
           "representation": "timepool [N, 26, 8, 1024]: mean over the 16x16 spatial tokens per tubelet (whole "
                             "frame, background included); points 0 = patch embedding, 1-24 = blocks, 25 = final LN",
           "pixel_reference": "artifacts/pixels grayscale 32x32 (8x8 block mean of 256^2), frames (2t, 2t+1) "
                              "averaged -> 8 x 1024; Cartesian = disk centroid averaged per frame pair",
           "clips": {"constvel": "direction set, motion == velocity (n=750, speed 1-7 m/s)",
                     "accel_direction_set": "direction set, motion == acceleration, from rest (n=750)",
                     "accel_acceleration_set": "acceleration set, from rest (n=1536)",
                     "reversed_*": "same clips, frames reversed (reversed accelerating clips decelerate to rest); "
                                   "pixel curvature is identical under reversal by construction",
                     "random_init_*": "random-init V-JEPA 2 ViT-L, same clips"},
           "null": {"isotropic": "random walk, each clip's own step norms, isotropic Gaussian directions in 1024-d",
                    "covmatched": "same step norms; directions ~ N(0, pooled second moment of the real "
                                  "displacements at that point) (the PCA random-walk null)"},
           "zone": ZONE, "n_boot": B, "seed": 0}

    curves = {}
    for k, C in lat.items():
        curves[k], _ = boot_mean(C, rng)
    out["latent_curvature_by_point"] = curves
    out["pixel_curvature"] = {k: {"mean": float(np.nanmean(v)), "ci95": boot_stat(lambda i, v=v: np.nanmean(v[i]),
                                                                                  len(v), rng)} for k, v in pix.items()}
    out["cartesian_curvature"] = {k: {"mean": float(np.nanmean(v)), "ci95": boot_stat(
        lambda i, v=v: np.nanmean(v[i]), len(v), rng)} for k, v in cart.items()}
    pix_for = {"constvel": pix["constvel"], "accel_direction_set": pix["accel_direction_set"],
               "accel_acceleration_set": pix["accel_acceleration_set"], "reversed_constvel": pix["constvel"],
               "reversed_accel_direction_set": pix["accel_direction_set"],
               "random_init_constvel": pix["constvel"], "random_init_accel_direction_set": pix["accel_direction_set"]}
    out["straightening_index_by_point"] = {k: boot_mean(pv[:, None] - lat[k], rng)[0] for k, pv in pix_for.items()}
    out["zone_test"] = {k: zone_test(lat[k], rng) for k in ["constvel", "accel_direction_set",
                                                           "accel_acceleration_set", "reversed_constvel",
                                                           "random_init_constvel"]}
    # reversed - forward, paired per clip
    out["reversed_minus_forward"] = {
        "constvel": boot_mean(lat["reversed_constvel"] - lat["constvel"], rng)[0],
        "accel_direction_set": boot_mean(lat["reversed_accel_direction_set"] - lat["accel_direction_set"], rng)[0]}

    # geometry links, per point
    rows = []
    acc_mag = dfa["acceleration_mps2"].to_numpy(float)
    for p in range(POINTS):
        c_v, c_a = lat["constvel"][:, p], lat["accel_direction_set"][:, p]
        n = len(c_v)
        r_sp = spearmanr(c_v, speed)[0]
        rows.append({
            "point": p,
            "constvel_spearman_curv_speed": float(r_sp),
            "constvel_spearman_curv_speed_ci95": boot_stat(lambda i: spearmanr(c_v[i], speed[i])[0], n, rng, 500),
            "constvel_partial_spearman_curv_speed_given_pixelcurv": partial_spearman(c_v, speed, pix["constvel"]),
            "accel_minus_constvel_mean_curv_deg": float(c_a.mean() - c_v.mean()),
            "accel_minus_constvel_ci95": boot_stat(
                lambda i: c_a[rng.integers(0, len(c_a), len(c_a))].mean() - c_v[i].mean(), n, rng, 500),
            "auroc_accel_vs_constvel": auroc(c_a, c_v),
            "accelset_spearman_curv_accel_magnitude": float(
                spearmanr(lat["accel_acceleration_set"][:, p], acc_mag)[0])})
    out["geometry_links_by_point"] = rows
    out["pixel_links"] = {"constvel_spearman_pixelcurv_speed": float(spearmanr(pix["constvel"], speed)[0]),
                          "auroc_pixelcurv_accel_vs_constvel": auroc(pix["accel_direction_set"], pix["constvel"])}
    pp = json.loads((RES / "p1a_perpatch_direction_vjepa2.json").read_text())
    dec = dict(zip(pp["points"], pp["curves"]["perpos_mean_r2"]))
    cm = np.array(curves["constvel"]["mean"])
    ks = sorted(dec)
    out["perpatch_decodability_link"] = {
        "source": "results/p1a_perpatch_direction_vjepa2.json curves.perpos_mean_r2 (direction, per position)",
        "points": ks, "perpatch_r2": [dec[k] for k in ks], "constvel_curvature": [float(cm[k]) for k in ks],
        "spearman_over_points_ge8": float(spearmanr([dec[k] for k in ks if k >= 8], [cm[k] for k in ks if k >= 8])[0]),
        "perpatch_argmax_point": int(max(dec, key=dec.get))}

    # speed-binned curves for the figure
    fig_pts = [0, 10, 25]
    out["speed_binned"] = {str(p): {str(int(s)): boot_stat(lambda i, s=s, p=p: lat["constvel"][speed == s, p][i].mean(),
                                                           int((speed == s).sum()), rng, 500)
                                    + [float(lat["constvel"][speed == s, p].mean())]
                                    for s in np.unique(speed)} for p in fig_pts}
    out["speed_binned_pixel"] = {str(int(s)): float(pix["constvel"][speed == s].mean()) for s in np.unique(speed)}
    out["wall_seconds"] = time.time() - t0
    (RES / "p5_straightening.json").write_text(json.dumps(out, indent=1))
    figure(out, fig_pts)
    print(f"done {time.time() - t0:.0f}s")


def figure(out, fig_pts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"constvel": "#2a78d6", "accel_acceleration_set": "#eb6834", "reversed_constvel": "#1baf7a",
           "random_init_constvel": "#8a8984", "null_covmatched_constvel": "#4a3aa7", "null_isotropic_constvel": "#e87ba4"}
    lab = {"constvel": "constant velocity", "accel_acceleration_set": "accelerating (acceleration set)",
           "reversed_constvel": "constant velocity, time-reversed", "random_init_constvel": "random-init encoder",
           "null_covmatched_constvel": "random walk, covariance-matched",
           "null_isotropic_constvel": "random walk, isotropic"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    a = ax[0]
    x = np.arange(POINTS)
    a.axvspan(7.5, 12.5, color="#eda100", alpha=0.13, lw=0, label="emergence zone (points 8-12)")
    for k in col:
        c = out["latent_curvature_by_point"][k]
        a.plot(x, c["mean"], color=col[k], lw=2, ls="--" if k.startswith("null") else "-", label=lab[k])
        a.fill_between(x, c["lo"], c["hi"], color=col[k], alpha=0.2, lw=0)
    for k, ls in [("constvel", ":"), ("accel_acceleration_set", "-.")]:
        a.axhline(out["pixel_curvature"][k]["mean"], color=col[k], lw=1.2, ls=ls,
                  label=f"pixels, {lab[k].split(' (')[0]}")
    a.set_xlabel("probe point (0 = patch embedding, 25 = final LN)")
    a.set_ylabel("curvature (deg, mean over clips)")
    a.set_title("Trajectory curvature by depth (95% bootstrap CI over clips)", fontsize=10)
    a.legend(fontsize=7, frameon=False, ncol=2, loc="best")
    a.spines[["top", "right"]].set_visible(False)
    b = ax[1]
    pc = ["#2a78d6", "#eb6834", "#1baf7a"]
    for c, p in zip(pc, fig_pts):
        d = out["speed_binned"][str(p)]
        s = np.array([int(k) for k in d])
        m = np.array([v[2] for v in d.values()])
        lo, hi = np.array([v[0] for v in d.values()]), np.array([v[1] for v in d.values()])
        rho = out["geometry_links_by_point"][p]["constvel_spearman_curv_speed"]
        b.errorbar(s, m, yerr=[m - lo, hi - m], color=c, marker="o", ms=6, lw=2, capsize=2,
                   label=f"point {p} (Spearman {rho:+.2f})")
    b.set_xlabel("speed (m/s), constant-velocity clips")
    b.set_ylabel("latent curvature (deg)")
    b.set_title("Curvature vs speed", fontsize=10)
    b.legend(fontsize=8, frameon=False)
    b.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig_straightening.png", dpi=150)


if __name__ == "__main__":
    main()

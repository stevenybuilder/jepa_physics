"""Time as a 4th variable, part 3 (CPU): what does the trained per-step time code do that a positional code cannot?

Speed set, per point, V-JEPA 2 vs the random-init copy (same clips: the random-init timepool subset if partial).
Per clip i, residual r_it = step feature - clip mean; time slope vector b_i = sum_t (t - 3.5) r_it / sum_t (t - 3.5)^2.
  shared_fraction        |mean_i b_i|^2 / mean_i |b_i|^2  (1 = every clip has the same time direction and size)
  mean_pairwise_cosine   mean cosine between clips' b_i
  band_cosines           cosine between the mean b of the slow / mid / fast speed bands
  step_share             fraction of held-out residual variance explained by knot step centroids (shared curve)
  time_probe_transfer    ridge t-probe on mid-band knot clips' per-step features, read on held-out slow / fast
                         clips: MAE (steps) and R^2
  orthogonality          |cos| between the shared time direction (mean b) and the speed probe axis, and the largest
                         principal cosine between the time subspace (top-3 PCs of the step centroids) and the direction
                         probe plane (sin, cos) / the speed axis; references: isotropic E|cos| = sqrt(2 / (pi D)) and
                         |cos| between the speed axis and 200 random clip-difference directions (data-shaped null).
Probes (speed, direction) are ridge on meanpool of probe clips at the same point, mapped to raw space (coef / scale).

  python scripts/run_time_positional.py --act-root /dev/shm/wm_p5 --out results/p5_time_positional.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from wm import timeman as tm
from wm.data import PROJECT_ROOT
from wm.p2_data import load_inputs
from wm.provenance import provenance

T = 8
BANDS = {"slow": (0.0, 1.0), "mid": (1.5, 2.75), "fast": (3.25, 9.0)}


def slope_vectors(R):
    tc = np.arange(T) - (T - 1) / 2
    return np.einsum("t,ntd->nd", tc, R) / (tc ** 2).sum()


def pairwise_cos(B, n_max=600, seed=0):
    rng = np.random.default_rng(seed)
    B = B[rng.permutation(len(B))[:n_max]]
    U = B / np.linalg.norm(B, axis=1, keepdims=True)
    G = U @ U.T
    return float((G.sum() - np.trace(G)) / (len(U) * (len(U) - 1)))


def cosabs(a, b):
    return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def raw_axis(X, y):
    m = tm.ridge(X, y)
    sc, rg = m.named_steps["standardscaler"], m.named_steps["ridgecv"]
    W = np.atleast_2d(rg.coef_) / sc.scale_[None]
    return W                                            # [n_targets, D]


def max_principal_cos(A, B):
    """Largest cosine between span(A rows) and span(B rows)."""
    qa, _ = np.linalg.qr(np.asarray(A, float).T)
    qb, _ = np.linalg.qr(np.asarray(B, float).T)
    return float(np.linalg.svd(qa.T @ qb, compute_uv=False)[0])


def one(path, layer, rows_ok, label):
    d = load_inputs("speed", layer)
    df = d["df"]
    v, th = df["speed_mps"].to_numpy(float), df["theta_degrees"].to_numpy(float)
    role = np.where(rows_ok, d["role"], "none")
    F = np.asarray(np.load(path, mmap_mode="r")[:, layer], np.float32)
    R = tm.remove_clip_mean(np.nan_to_num(F))
    band = np.full(len(v), "", object)
    for k, (lo, hi) in BANDS.items():
        band[(v >= lo) & (v <= hi)] = k
    ok = rows_ok
    B = slope_vectors(R[ok])
    mb = B.mean(0)
    out = {"n_clips": int(ok.sum()),
           "shared_fraction": float((mb ** 2).sum() / (B ** 2).sum(1).mean()),
           "mean_pairwise_cosine": pairwise_cos(B),
           "slope_norm_median": float(np.median(np.linalg.norm(B, axis=1)))}
    bm = {k: slope_vectors(R[ok & (band == k)]).mean(0) for k in BANDS}
    out["band_cosines"] = {"slow_mid": cosabs(bm["slow"], bm["mid"]), "slow_fast": cosabs(bm["slow"], bm["fast"]),
                           "mid_fast": cosabs(bm["mid"], bm["fast"])}
    knot, held = ok & (role == "knot"), ok & np.isin(role, ["probe", "test"])
    C = np.stack([R[knot][:, t].mean(0) for t in range(T)])
    Rh = R[held]
    out["step_share"] = float(1 - ((Rh - C[None]) ** 2).sum() / (Rh ** 2).sum())
    mid_k = knot & (band == "mid")
    tp = tm.ridge(F[mid_k].reshape(-1, F.shape[-1]), np.tile(np.arange(T), mid_k.sum()))
    out["time_probe_transfer"] = {}
    for k in BANDS:
        r = held & (band == k)
        p = tp.predict(F[r].reshape(-1, F.shape[-1]))
        y = np.tile(np.arange(T), r.sum())
        out["time_probe_transfer"][k] = {"mae_steps": float(np.abs(p - y).mean()), "r2": tm.r2(y, p),
                                         "bias_steps": float((p - y).mean()), "n_clips": int(r.sum())}
    # orthogonality against meanpool probes at the same point (probe clips, trained encoder's own meanpool when label
    # is vjepa2; the random encoder's meanpool = step mean of its timepool)
    X = F.mean(1) if label == "random" else d["X"]
    pr = ok & (role == "probe")
    ax_speed = raw_axis(X[pr], v[pr])[0]
    ax_dir = raw_axis(X[pr], np.stack([np.sin(np.radians(th[pr])), np.cos(np.radians(th[pr]))], 1))
    Cc = C - C.mean(0)
    time_sub = np.linalg.svd(Cc, full_matrices=False)[2][:3]
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(ok)
    diffs = X[rng.choice(idx, 200)] - X[rng.choice(idx, 200)]
    Dm = X.shape[1]
    out["orthogonality"] = {
        "cos_mean_time_dir_vs_speed_axis": cosabs(mb, ax_speed),
        "maxcos_mean_time_dir_vs_direction_plane": max_principal_cos(mb[None], ax_dir),
        "maxcos_time_subspace3_vs_speed_axis": max_principal_cos(time_sub, ax_speed[None]),
        "maxcos_time_subspace3_vs_direction_plane": max_principal_cos(time_sub, ax_dir),
        "isotropic_expected_abs_cos": float(np.sqrt(2 / (np.pi * Dm))),
        "null_speed_axis_vs_random_clip_differences_abs_cos_mean": float(np.mean([cosabs(ax_speed, q) for q in diffs])),
        "null_speed_axis_vs_random_clip_differences_abs_cos_p95": float(np.percentile([cosabs(ax_speed, q) for q in diffs], 95)),
    }
    return label, layer, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-root", default=str(PROJECT_ROOT / "artifacts" / "activations"))
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 19, 22])
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "p5_time_positional.json"))
    args = ap.parse_args()
    t0 = time.time()
    root = Path(args.act_root) / "speed"
    ids = json.loads((PROJECT_ROOT / "artifacts/activations/speed/vjepa2/ids.json").read_text())
    sub_f = root / "random" / "ids_available.json"
    sub = set(json.loads(sub_f.read_text())) if sub_f.exists() else set(ids)
    rows_sub = np.array([i in sub for i in ids])
    cfg = [("vjepa2_all", root / "vjepa2" / "timepool.npy", np.ones(len(ids), bool)),
           ("vjepa2_subset", root / "vjepa2" / "timepool.npy", rows_sub),
           ("random", root / "random" / "timepool.npy", rows_sub)]
    outs = Parallel(n_jobs=args.jobs)(delayed(one)(p, l, r, lab) for lab, p, r in cfg for l in args.layers)
    res = {}
    for lab, l, o in outs:
        res.setdefault(lab, {})[str(l)] = o
    res["n_random_subset_clips"] = int(rows_sub.sum())
    res["keys"] = {"<encoder>[point]": __doc__.split("\n\n")[1]}
    res["provenance"] = provenance(seeds={"all": 0}, layers=args.layers, act_root=args.act_root,
                                   wall_s=round(time.time() - t0, 1))
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("wrote", args.out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

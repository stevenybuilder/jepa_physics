"""Ring or velocity plane? (spec.md section 6; lit-review item 5). Per layer, on train clips:

  1. direction set: Procrustes R^2 of the 64 direction centroids onto (cos theta, sin theta);
  2. speed set (crosses 64 speeds with direction): Procrustes R^2 of speed-bin x direction-bin cell centroids onto
     (cos theta, sin theta) vs (v cos theta, v sin theta), and the ring radius per speed bin;
  3. speed readout along a chord steer (speed set, test clips): steer direction by d_theta along the straight chord
     between direction centroids (PCA-k, additive) and read speed at the midpoint vs the start with (a) a ridge speed
     probe on probe-fold clips and (b) the radius in the fitted velocity plane. A linear velocity plane predicts a
     midpoint ratio of cos(d_theta / 2); a ring with a separate speed code predicts 1.

Writes results/p2_velocity_plane.json (one entry per layer) and figures/fig4_ring_radius_vs_speed.png.

  python scripts/run_velocity_plane.py --layers 4 8 12 16 20
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wm import geometry_checks as gc
from wm import manifold as mf
from wm.data import PROJECT_ROOT
from wm.p2_data import load_inputs
from wm.provenance import layer_role, provenance

SHIFTS = (45.0, 90.0, 135.0, 180.0)


def chord_speed_readout(s, k, n_clips, seed):
    """Midpoint / start ratio of two speed readouts along a direction chord steer on the speed set."""
    tr, probe = s["is_train"], s["role"] == "probe"
    speed = s["df"]["speed_mps"].to_numpy(float)
    pca = mf.fit_pca(s["X"][tr], k)
    cent = mf.centroids(pca.project(s["X"][tr]), s["y"][tr])            # direction centroids, averaged over speed
    curve = mf.fit_curve(cent, True, angle="labels")
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 15))).fit(s["X"][probe], speed[probe])
    vel = gc.velocity_plane_check(s["X"][tr], s["y"][tr], speed[tr])["_velocity_fit"]
    plane_radius = lambda X: np.linalg.norm((X - vel["mu_y"]) @ vel["R"] / vel["scale"] + vel["mu_t"], axis=1)
    rng = np.random.default_rng(seed)
    test = np.flatnonzero(s["role"] == "test")
    pick = rng.choice(test, size=min(n_clips, len(test)), replace=False)
    x, th = s["X"][pick].astype(float), s["y"][pick]
    Z, resid = pca.project(x), pca.complement(x)
    out = []
    for dth in SHIFTS:
        tgt = (th + dth) % 360.0
        Zk = mf.linear_coords(Z, mf.piecewise_linear_point(curve, th), mf.piecewise_linear_point(curve, tgt), 3)
        start, mid = x, pca.lift(Zk[:, 1]) + resid
        r_ridge = ridge.predict(mid) / np.maximum(ridge.predict(start), 1e-6)
        r_plane = plane_radius(mid) / np.maximum(plane_radius(start), 1e-6)
        out.append({"d_theta": dth, "velocity_plane_prediction": float(np.cos(np.radians(dth) / 2)),
                    "ratio_ridge_speed_probe": float(np.median(r_ridge)),
                    "ratio_velocity_plane_radius": float(np.median(r_plane)), "n": len(pick)})
    return out


def run(args):
    res = {"provenance": provenance(args.speed_split, seeds={"seed": args.seed}, layers=args.layers, pool="meanpool",
                                    k=args.k), "layers": {}}
    for layer in args.layers:
        d = load_inputs("direction", layer, "direction", args.dir_act_dir, args.dir_table, args.dir_split)
        s = load_inputs("speed", layer, "direction", args.speed_act_dir, args.speed_table, args.speed_split)
        tr = d["is_train"]
        C = np.array([d["X"][tr & (d["y"] == v)].mean(0) for v in np.unique(d["y"][tr])])
        r = np.radians(np.unique(d["y"][tr]))
        vp = gc.velocity_plane_check(s["X"][s["is_train"]], s["y"][s["is_train"]],
                                     s["df"]["speed_mps"].to_numpy(float)[s["is_train"]])
        vp.pop("_velocity_fit")
        res["layers"][str(layer)] = {
            **layer_role("direction", layer, "direction"),
            "direction_set_procrustes_r2_ring": gc.procrustes_r2(C, np.stack([np.cos(r), np.sin(r)], 1))["r2"],
            "speed_set": vp, "chord_speed_readout": chord_speed_readout(s, args.k, args.n_clips, args.seed)}
        print(f"layer {layer}: ring R2 {vp['procrustes_r2_ring']:.3f}  velocity R2 {vp['procrustes_r2_velocity']:.3f}"
              f"  radius top/bottom {vp['radius_ratio_top_bottom']:.2f} (speed ratio {vp['speed_ratio_top_bottom']:.1f})")
    res["note"] = ("velocity plane: R2_velocity > R2_ring, radius ratio ~ speed ratio, chord midpoint ratio ~ "
                   "cos(d_theta/2); ring: the reverse, radius flat, ratio ~ 1")
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.results_dir) / "p2_velocity_plane.json").write_text(json.dumps(res, indent=1))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for layer, r in res["layers"].items():
        rb = r["speed_set"]["radius_by_speed"]
        ax.plot([x["speed_mean"] for x in rb], [x["radius"] for x in rb], marker="o", label=f"layer {layer}")
    ax.set(xlabel="speed (m/s, bin mean)", ylabel="ring radius (activation units)",
           title="Direction ring radius vs speed (speed set)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(args.figures_dir) / "fig4_ring_radius_vs_speed.png", dpi=150)
    plt.close(fig)
    return res


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--n-clips", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    for ds in ("dir", "speed"):
        p.add_argument(f"--{ds}-act-dir", default=None)
        p.add_argument(f"--{ds}-table", default=None)
        p.add_argument(f"--{ds}-split", default=None)
    p.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--figures-dir", default=str(PROJECT_ROOT / "figures"))
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse())

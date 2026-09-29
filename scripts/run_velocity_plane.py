"""Ring or velocity plane? (project spec section 6; literature review item 5). Per layer, on train clips:

  1. direction set: Procrustes R^2 of the 64 direction centroids onto (cos theta, sin theta);
  2. speed set (crosses 64 speeds with direction): Procrustes R^2 of speed-bin x direction-bin cell centroids onto
     (cos theta, sin theta) vs (v cos theta, v sin theta), and the ring radius per speed bin;
  3. speed readout along a chord steer (speed set, test clips): steer direction by d_theta along the straight chord
     between direction centroids (PCA-k, additive) and read speed at the midpoint with readouts that are not linear
     in the edit (MLP, nearest-real at same vs reduced speed, Eq. 9 speed distribution). A velocity plane predicts
     the midpoint reads ~v cos(d_theta / 2); a ring with a separate speed code predicts it unchanged.

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

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))  # run without installing wm
from wm import geometry_checks as gc
from wm import manifold as mf
from wm.data import PROJECT_ROOT
from wm.p2_data import load_inputs
from wm.provenance import layer_role, provenance

SHIFTS = (45.0, 90.0, 135.0, 180.0)


PREDICTIONS = {
    "mlp_speed_ratio": "MLP speed read at the chord midpoint / at the start. ring: ~1 (speed code untouched); "
                       "velocity plane: ~cos(d_theta/2)",
    "nearest_real_same_minus_reduced": "agreement R of the midpoint with real clips at the midpoint direction and the "
                                       "SAME speed minus R with real clips at speed v*cos(d_theta/2). ring: > 0; "
                                       "velocity plane: < 0",
    "eq9_speed_mean_ratio": "expected speed under the Eq. 9 distribution over speed-bin x direction-bin cell "
                            "centroids, marginalised over direction, midpoint / start. ring: ~1; velocity plane: "
                            "~cos(d_theta/2). (Plain speed centroids average direction out and collapse to the "
                            "centre in a velocity plane, so they cannot see speed there.)",
    "eq9_speed_entropy_change": "Eq. 9 speed-distribution entropy, midpoint minus start (reported; no sharp "
                                "prediction)",
}


def chord_speed_readout(s, k, n_clips, seed):
    """Speed readouts that are NOT linear in the edit, along a direction chord steer on the speed set.

    The chord edit is linear, so a linear speed probe reads ~1 and the norm of a linear plane map reads
    ~cos(d_theta/2) by construction; neither can tell a ring from a velocity plane. Readouts here: (a) an MLP speed
    evaluator, (b) nearest-real agreement at the midpoint against real clips at the midpoint direction with the same
    speed vs the reduced speed v cos(d_theta/2), (c) the Eq. 9 speed distribution (softmax over distances to the 64
    speed centroids): its mean and entropy. Same role split as run_part2: the chord's PCA and direction centroids use
    knot folds 0-2; the MLP and the Eq. 9 speed centroids use probe folds 3-4; steered clips and the real clips the
    agreement compares with are test (a clip is never compared with itself)."""
    from wm.bakeoff import MLPReadout
    tr, probe = s["role"] == "knot", s["role"] == "probe"
    speed = s["df"]["speed_mps"].to_numpy(float)
    pca = mf.fit_pca(s["X"][tr], k)
    cent = mf.centroids(pca.project(s["X"][tr]), s["y"][tr])            # direction centroids, averaged over speed
    curve = mf.fit_curve(cent, True, angle="labels")
    mlp = MLPReadout(s["X"][probe], speed[probe], periodic=False, seed=seed)
    cell, cell_speed = speed_direction_cells(speed, s["y"], tr)
    eq9 = mf.BehaviourManifold(s["X"][probe], cell[probe], False, mode="centroid_sq")
    eq9_speed = np.array([cell_speed[int(c)] for c in eq9.values])
    rng = np.random.default_rng(seed)
    test = np.flatnonzero(s["role"] == "test")
    pick = rng.choice(test, size=min(n_clips, len(test)), replace=False)
    x, th, v = s["X"][pick].astype(float), s["y"][pick], speed[pick]
    Z, resid = pca.project(x), pca.complement(x)
    Xt, tht, vt = s["X"][test], s["y"][test], speed[test]
    v_tol = (speed.max() - speed.min()) / 16

    def real_mean(i, theta, vel):
        """Mean of real test clips within 11.25 deg of theta and v_tol of vel (excluding the steered clip)."""
        ok = (mf.value_error(tht, theta, True) <= 11.25) & (np.abs(vt - vel) <= v_tol) & (test != pick[i])
        return Xt[ok].mean(0) if ok.sum() >= 2 else None

    def eq9_mean_speed(X):
        return eq9.value_distribution(X) @ eq9_speed

    xt_all = s["X"][test]
    mlp_r2 = float(1 - ((mlp.predict(xt_all) - speed[test]) ** 2).sum() / ((speed[test] - speed[test].mean()) ** 2).sum())
    e9 = eq9_mean_speed(xt_all)
    eq9_corr = float(np.corrcoef(e9, speed[test])[0, 1])

    out = []
    for dth in SHIFTS:
        tgt = (th + dth) % 360.0
        Zk = mf.linear_coords(Z, mf.piecewise_linear_point(curve, th), mf.piecewise_linear_point(curve, tgt), 3)
        start, mid = x, pca.lift(Zk[:, 1]) + resid
        th_mid = (th + dth / 2) % 360.0
        diffs = []
        for i in range(len(pick)):
            same, reduced = real_mean(i, th_mid[i], v[i]), real_mean(i, th_mid[i], v[i] * np.cos(np.radians(dth) / 2))
            if same is None or reduced is None:
                continue
            R = lambda ref: 1 - np.linalg.norm(mid[i] - ref) / max(np.linalg.norm(start[i] - ref), 1e-12)
            diffs.append(R(same) - R(reduced))
        out.append({"d_theta": dth, "velocity_plane_prediction_ratio": float(np.cos(np.radians(dth) / 2)),
                    "mlp_speed_ratio": float(np.median(mlp.predict(mid) / np.maximum(mlp.predict(start), 1e-6))),
                    "nearest_real_same_minus_reduced": float(np.median(diffs)) if diffs else None,
                    "n_nearest_real": len(diffs),
                    "eq9_speed_mean_ratio": float(np.median(eq9_mean_speed(mid) / eq9_mean_speed(start))),
                    "eq9_speed_entropy_change": float(np.median(eq9.entropy(mid) - eq9.entropy(start))),
                    "n": len(pick), "mlp_r2_unsteered_test": mlp_r2, "eq9_speed_corr_unsteered_test": eq9_corr})
    return out


def speed_direction_cells(speed, theta, fit_rows, n_speed=8, n_dir=16):
    """Cell id per clip (speed octile x 22.5-degree direction bin) and each cell's speed-bin mean speed. Octile
    edges and bin mean speeds come from `fit_rows` only (knot folds), never from test clips."""
    edges = np.quantile(speed[fit_rows], np.linspace(0, 1, n_speed + 1)[1:-1])
    sb = np.clip(np.searchsorted(edges, speed, side="right"), 0, n_speed - 1)
    db = np.floor(np.asarray(theta) / (360.0 / n_dir)).astype(int) % n_dir
    cell = (sb * n_dir + db).astype(float)
    fit_mean = {b: float(speed[fit_rows & (sb == b)].mean()) if (fit_rows & (sb == b)).any() else float(speed[sb == b].mean())
                for b in np.unique(sb)}
    return cell, {int(c): fit_mean[int(c) // n_dir] for c in np.unique(cell)}


def chord_verdict(rows):
    """Rule-based reading at the largest d_theta: each readout says ring, velocity_plane or inconclusive."""
    r = max(rows, key=lambda q: q["d_theta"])
    c = r["velocity_plane_prediction_ratio"]
    def ratio_call(x):
        if x is None:
            return "inconclusive"
        return "ring" if abs(x - 1) < 0.5 * (1 - c) else "velocity_plane" if abs(x - c) < 0.5 * (1 - c) else "inconclusive"
    have = [q for q in rows if q["nearest_real_same_minus_reduced"] is not None]    # at 180 deg v*cos -> 0: no clips
    nr_row = max(have, key=lambda q: q["d_theta"]) if have else None
    nr = nr_row["nearest_real_same_minus_reduced"] if nr_row else None
    calls = {"mlp_speed_ratio": (ratio_call(r["mlp_speed_ratio"]) if r["mlp_r2_unsteered_test"] >= 0.5
                                 else "inconclusive (MLP reads speed poorly on real clips)"),
             "eq9_speed_mean_ratio": (ratio_call(r["eq9_speed_mean_ratio"]) if r["eq9_speed_corr_unsteered_test"] >= 0.5
                                      else "inconclusive (Eq. 9 speed read poorly on real clips)"),
             "nearest_real_same_minus_reduced": ("inconclusive" if nr is None or abs(nr) < 0.02
                                                 else "ring" if nr > 0 else "velocity_plane")}
    decided = {v for v in calls.values() if not v.startswith("inconclusive")}
    summary = (f"chord readouts agree: {decided.pop()}" if len(decided) == 1 else
               "chord readouts disagree; rely on Procrustes R2 and radius-vs-speed" if decided else
               "no chord readout separates ring from velocity plane; the velocity-plane evidence is the Procrustes "
               "R2 comparison and the ring radius vs speed")
    return {"d_theta": r["d_theta"], "d_theta_nearest_real": nr_row["d_theta"] if nr_row else None, "calls": calls,
            "summary": summary}


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
            "speed_set": vp, "chord_speed_readout": (chord := chord_speed_readout(s, args.k, args.n_clips, args.seed)),
            "chord_verdict": chord_verdict(chord), "chord_predictions": PREDICTIONS,
            "chord_roles": "chord fit on knot folds 0-2; MLP and Eq. 9 speed centroids on probe folds 3-4; steered "
                           "and nearest-real clips from test"}
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

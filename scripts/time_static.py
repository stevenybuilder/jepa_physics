"""Static-disk control for the time code: frame counter or odometer?

At zero speed the distance travelled is 0 at every step, so an odometer-like time code predicts no advance, while a
frame counter (token-time position code) advances one step per step regardless of motion.

  render  (Mac, CPU)  96 non-knot speed-set clips; for each, a STATIC twin (speed 0, acceleration 0, same start,
                      direction, background, renderer) and a MOVING twin (unchanged metadata) as the renderer-matched
                      reference -> artifacts/time_static/{static,moving}/clip_XXXX.mp4, plan.json; renderer IoU check
                      (render_twin.validate) and a static smoke check (disk centroid spread over the 16 frames).
  extract (box, GPU)  V-JEPA 2 and the random-init copy (torch.manual_seed(0), same as every other random-init array):
                      timepool [n, 26, 8, D] of both twin sets -> artifacts/time_static/feat_{model}.npz
  score   (CPU, where the stored speed-set timepools live)  per model and point, fit on MOVING knot clips (folds 0-2)
                      exactly as run_time_manifold / run_time_positional (residual = step - clip mean, PCA-64, step
                      centroids, smoothing spline, mean time-slope direction, mid-band time probe), then read the
                      static twins, the moving twins and the originals' stored features.

  python scripts/time_static.py render|extract|score [--out artifacts/time_static] [--act-root ...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import PROJECT_ROOT, decode, disk_pixels, frame_hash, load_table  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "time_static"
POINTS = (12, 22)
T = 8
BANDS = {"slow": (0.0, 1.0), "mid": (1.5, 2.75), "fast": (3.25, 9.0)}


def render(args):
    from wm.p2_data import load_inputs
    from wm.render_twin import load_meta, validate, write_twin
    out = Path(args.out)
    d0 = load_inputs("speed", 0)
    df, role = d0["df"], d0["role"]
    rng = np.random.default_rng(0)
    pool = np.flatnonzero(role != "knot")
    rows = np.sort(rng.choice(pool, args.n, replace=False))
    spread, hashes = [], {"static": [], "moving": []}
    for k, r in enumerate(rows):
        meta = load_meta("speed", int(df["id"].iloc[r]))
        for kind, m in (("static", {**meta, "speed_mps": 0.0, "acceleration_mps2": 0.0}), ("moving", meta)):
            fr = write_twin(m, None, out / kind / f"clip_{k:04d}.mp4")
            hashes[kind].append(frame_hash(fr))
            if kind == "static":
                pix = disk_pixels(fr)
                c = np.array([np.argwhere(p).mean(0) for p in pix])
                spread.append(float(np.abs(c - c.mean(0)).max()))
    val = validate("speed", n=20, seed=0)
    info = {"rows": rows.tolist(), "ids": df["id"].to_numpy()[rows].tolist(), "n": int(len(rows)),
            "speeds": df["speed_mps"].to_numpy()[rows].tolist(), "roles": role[rows].tolist(),
            "static_smoke": {"max_centroid_deviation_px_over_16_frames_max": float(np.max(spread)),
                             "median": float(np.median(spread))},
            "renderer_validation_speed": {k: val[k] for k in ("codec", "mask_twin_iou_mean", "verdict") if k in val},
            "frame_hash": hashes}
    (out / "plan.json").write_text(json.dumps(info, indent=1))
    print("rendered", len(rows), info["static_smoke"], val.get("codec", {}).get("iou_mean"))


def extract(args):
    import torch
    from wm.extract import disk_mask, encode, load_model, pick_device, pool, preprocess, set_precision
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from time_predictor import gpu_lock
    out = Path(args.out)
    info = json.loads((out / "plan.json").read_text())
    set_precision()
    dev = pick_device()
    frames = {k: [decode(out / k / f"clip_{i:04d}.mp4") for i in range(info["n"])] for k in ("static", "moving")}
    for k in frames:
        assert [frame_hash(f) for f in frames[k]] == info["frame_hash"][k]
    for model_kind in ("vjepa2", "random"):
        model = load_model(model_kind, dev)
        res = {}
        with gpu_lock() as waited, torch.no_grad():
            t0 = time.time()
            for k, fl in frames.items():
                tp = []
                for s in range(0, len(fl), 8):
                    masks = torch.from_numpy(np.stack([disk_mask(f)[0] for f in fl[s:s + 8]])).to(dev)
                    points, _ = encode(model, preprocess(fl[s:s + 8]).to(dev))
                    tp.append(pool(points, masks)[1].astype(np.float16))
                res[k] = np.concatenate(tp)
            secs = time.time() - t0
        np.savez(out / f"feat_{model_kind}.npz", seconds=secs, lock_wait_s=waited, device=str(dev), **res)
        print(model_kind, secs, flush=True)
        del model
        torch.cuda.empty_cache()


def score(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wm import manifold as mf
    from wm import timeman as tm
    from wm.p2_data import load_inputs
    from wm.provenance import provenance
    out = Path(args.out)
    info = json.loads((out / "plan.json").read_text())
    rows = np.array(info["rows"])
    d0 = load_inputs("speed", 0)
    df, role = d0["df"], d0["role"]
    v = df["speed_mps"].to_numpy(float)
    knot = role == "knot"
    mid_knot = knot & (v >= 1.5) & (v <= 2.75)
    tt = np.arange(T, dtype=float)
    res = {"n_clips": int(len(rows)), "render": {k: info[k] for k in ("static_smoke", "renderer_validation_speed")},
           "models": {}}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for mi, model in enumerate(("vjepa2", "random")):
        feat = np.load(out / f"feat_{model}.npz")
        stored = np.load(Path(args.act_root) / "speed" / model / "timepool.npy", mmap_mode="r")
        res["models"][model] = {"extract_seconds": float(feat["seconds"]), "device": str(feat["device"])}
        for pi, L in enumerate(POINTS):
            F = np.asarray(stored[:, L], np.float64)
            R = tm.remove_clip_mean(F)
            Rk = R[knot].reshape(-1, R.shape[-1])
            pca = mf.fit_pca(Rk, 64)
            lab = np.tile(np.arange(T), knot.sum())
            cent = mf.centroids(pca.project(Rk), lab)
            C = cent["C"]
            curve = mf.fit_curve(cent, False, spline="smooth", extend="linear")
            gt = np.linspace(-6, 14, 4001)
            gp = curve(gt)
            tc = tt - tt.mean()
            mb = np.einsum("t,ntd->d", tc, R[knot]) / (tc ** 2).sum() / knot.sum()     # mean time-slope direction
            tprobe = tm.ridge(F[mid_knot].reshape(-1, F.shape[-1]), np.tile(np.arange(T), mid_knot.sum()))
            sets = {"static": np.asarray(feat["static"][:, L], np.float64),
                    "moving_twin": np.asarray(feat["moving"][:, L], np.float64),
                    "original_stored": F[rows]}
            m = {}
            per_clip = {}
            for name, X in sets.items():
                Rx = tm.remove_clip_mean(X)
                Z = pca.project(Rx)
                c_dir = Rx @ mb / (mb @ mb)                                       # [n, 8] coordinate along mean dir
                c_sp = tm.nearest_coord(Z, gt, gp)
                p_t = tprobe.predict(X.reshape(-1, X.shape[-1])).reshape(len(X), T)
                per_clip[name] = {"dir": tm.per_clip_slope(c_dir), "spline": tm.per_clip_slope(c_sp),
                                  "probe": tm.per_clip_slope(p_t)}
                share = 1 - ((Z - C[None]) ** 2).sum() / (Z ** 2).sum()
                full_share = 1 - ((Rx - pca.lift_delta(C)[None]) ** 2).sum() / (Rx ** 2).sum()
                m[name] = {"slope_mean_direction": tm.cluster_bootstrap(per_clip[name]["dir"]),
                           "slope_spline_coordinate": tm.cluster_bootstrap(per_clip[name]["spline"]),
                           "slope_mid_band_time_probe": tm.cluster_bootstrap(per_clip[name]["probe"]),
                           "probe_mae_steps": float(np.abs(p_t - tt).mean()),
                           "shared_curve_share_pca64": float(share), "shared_curve_share_full": float(full_share),
                           "within_clip_rms": float(np.sqrt((Rx ** 2).sum(-1).mean())),
                           "mean_coord_per_step_spline": tm.nearest_coord(Z, gt, gp).mean(0).tolist()}
                if name == "static":
                    axes[mi, pi].plot(tt, m[name]["mean_coord_per_step_spline"], "o-", label="static twin")
                else:
                    axes[mi, pi].plot(tt, m[name]["mean_coord_per_step_spline"], "o-", label=name.replace("_", " "))
            # advance of the static twins in units of the paired moving references (1 = frame counter, 0 = odometer)
            for ref in ("moving_twin", "original_stored"):
                m[f"static_over_{ref}"] = {
                    q: {"ratio_of_means": float(per_clip["static"][q].mean() / per_clip[ref][q].mean()),
                        "ci95": [float(x) for x in np.percentile(
                            [(lambda i: per_clip["static"][q][i].mean() / per_clip[ref][q][i].mean())(
                                np.random.default_rng(b).integers(0, len(rows), len(rows))) for b in range(1000)],
                            [2.5, 97.5])]}
                    for q in ("dir", "spline", "probe")}
            # on-manifold? step-mean (clip) features and per-step features vs the stored moving clips
            others = np.setdiff1d(np.arange(len(F)), rows)
            Mm = F[others].mean(1)
            nn = lambda A: mf.min_distance(A, Mm)                                                        # noqa: E731
            d_static, d_mov = nn(sets["static"].mean(1)), nn(sets["moving_twin"].mean(1))
            d_orig = nn(F[rows].mean(1))
            step_nn = lambda A: np.mean([mf.min_distance(A[:, t], F[others][:, t]).mean() for t in range(T)])  # noqa: E731
            m["distance_to_moving_clips"] = {
                "clip_mean_feature_nn_dist": {"static": float(d_static.mean()), "moving_twin": float(d_mov.mean()),
                                              "original_stored": float(d_orig.mean()),
                                              "static_over_moving_twin": tm.cluster_bootstrap(d_static / d_mov)},
                "per_step_feature_nn_dist_mean": {"static": float(step_nn(sets["static"])),
                                                  "moving_twin": float(step_nn(sets["moving_twin"])),
                                                  "original_stored": float(step_nn(F[rows]))},
                "moving_twin_vs_original_rel_diff": float(np.linalg.norm(sets["moving_twin"] - F[rows], axis=-1).mean()
                                                          / np.linalg.norm(F[rows], axis=-1).mean())}
            res["models"][model][str(L)] = m
            ax = axes[mi, pi]
            ax.plot(tt, tt, "k:", lw=1, label="frame counter (1 step/step)")
            ax.axhline(np.mean(m["moving_twin"]["mean_coord_per_step_spline"]), color="gray", lw=0.8, ls="--",
                       label="odometer at v = 0 (no advance)")
            ax.set_title(f"{model}, point {L}: static advance {m['static_over_moving_twin']['spline']['ratio_of_means']:.2f} "
                         f"x moving", fontsize=9)
            ax.set_xlabel("true step"); ax.set_ylabel("time-spline coordinate (moving knot fit)")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.fig, dpi=130)
    res["keys"] = {
        "models[m][point][set]": "set = static (speed-0 twins), moving_twin (renderer twins with the original metadata), "
                                 "original_stored (the originals' stored timepool). slope_*: per-clip OLS slope over the 8 "
                                 "steps of (dir) the projection on the moving knot clips' mean time-slope direction, "
                                 "(spline) the nearest-point coordinate on the moving knot step-centroid spline, (probe) the "
                                 "mid-band (1.5-2.75 m/s) knot time probe's predicted step; mean + 95% clip bootstrap. "
                                 "shared_curve_share: 1 - sum|R - C_t|^2 / sum|R|^2 of the set's within-clip residuals "
                                 "against the moving knot step centroids (PCA-64 / full); within_clip_rms: size of the "
                                 "within-clip (over-time) variation.",
        "models[m][point].static_over_<ref>": "static slope / paired reference slope (same 96 clips), ratio of means with "
                                              "a clip-bootstrap CI: 1.0 = frame counter, 0 = odometer",
        "models[m][point].distance_to_moving_clips": "nearest-neighbour distance (raw space) of each set's clip-mean and "
                                                     "per-step features to the other 1440 stored moving clips",
        "render": "static_smoke = max deviation of the disk's pixel centroid over the 16 frames (px); renderer validation "
                  "of re-rendered supplied speed clips vs the decoded originals"}
    res["provenance"] = provenance(seeds={"clips": 0, "bootstrap": 0}, points=list(POINTS), act_root=args.act_root,
                                   clip_ids=info["ids"])
    Path(args.result).write_text(json.dumps(res, indent=1))
    print("wrote", args.result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["render", "extract", "score"])
    ap.add_argument("--out", default=str(ART))
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--act-root", default=str(PROJECT_ROOT / "artifacts" / "activations"))
    ap.add_argument("--result", default=str(PROJECT_ROOT / "results" / "p5_time_static_control.json"))
    ap.add_argument("--fig", default=str(PROJECT_ROOT / "figures" / "fig_time_static_control.png"))
    args = ap.parse_args()
    {"render": render, "extract": extract, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()

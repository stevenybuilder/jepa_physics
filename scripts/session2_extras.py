"""Session 2 extras, CPU, local, from the pulled forward outputs:

  future-position audit  a ridge probe from the real per-tubelet pooled final-LN token (timepool[:, 25, t]) to the disk's
                         pixel centroid in that tubelet (decoded frames, both frames of the tubelet), fit on probe clips;
                         applied to the predictor's pooled forecasts (tubelets 4-7) for source, twin and every edit arm.
                         Ground truth for source and twin = pixel centroid of the decoded source clip / rendered twin MP4.
  time-reversed speed    speed probe (velocity clips only; a reversed accelerating clip is a decelerating clip) fit on
                         forward probe clips, read on forward vs reversed test clips at every point.
-> results/session2_future_position.json, results/session2_timerev_speed.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import PROJECT_ROOT, decode, disk_pixels, load_table  # noqa: E402
from wm.p2_data import load_inputs  # noqa: E402
from wm.probes import Standardizer, cv_select_alpha, fit_ridge, predict, score  # noqa: E402
from wm.probes import write_json  # noqa: E402

ACT = PROJECT_ROOT / "artifacts" / "activations" / "direction"
FWD = PROJECT_ROOT / "artifacts" / "session2" / "forward"
RES = PROJECT_ROOT / "results"
ARMS = ("probe_qr", "radius_matched", "spline", "spline_smooth", "chord", "random_matched")
LAYERS = (2, 8, 12, 22)


def centroids(frames):
    """[8, 2] pixel centroid (x, y) per tubelet; NaN where the disk is absent in both frames."""
    pix = disk_pixels(frames)                                        # [16, 256, 256] bool
    out = np.full((8, 2), np.nan)
    for t in range(8):
        yy, xx = np.nonzero(pix[2 * t:2 * t + 2].any(0))
        if len(xx):
            out[t] = xx.mean(), yy.mean()
    return out


def boot(v, n=1000, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    m = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(1)
    return {"mean": float(v.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], "n": int(len(v))}


def future_position():
    df = load_table("direction")
    d0 = load_inputs("direction", 0)
    probe, test = d0["role"] == "probe", d0["role"] == "test"
    folds = d0["fold"][probe]
    cache = FWD.parent / "centroids_direction.npy"
    if cache.exists():
        cen = np.load(cache)
    else:
        cen = np.stack([centroids(decode(v)) for v in df["video"]])  # [1500, 8, 2]
        np.save(cache, cen)
    tp = np.load(ACT / "vjepa2" / "timepool.npy", mmap_mode="r")
    probes = {}
    real_test = {}
    for t in range(4, 8):
        X = np.asarray(tp[:, 25, t], np.float64)
        ok = np.isfinite(cen[:, t]).all(1)
        m = probe & ok
        st = Standardizer().fit(X[m])
        cv = cv_select_alpha(st.transform(X[m]), cen[m, t], folds[ok[probe]], score_fn=lambda a, b: score(a, b, "linear"))
        W, b = fit_ridge(st.transform(X[m]), cen[m, t], cv["alpha"])
        probes[t] = (st, W, b)
        mt = test & ok
        err = np.linalg.norm(predict(st.transform(X[mt]), W, b) - cen[mt, t], axis=1)
        real_test[str(t)] = {"alpha": cv["alpha"], "cv_r2": cv["cv_mean"], "test_err_px_mean": float(err.mean()),
                             "test_err_px_median": float(np.median(err)), "n_test": int(mt.sum())}

    def read(Z):                                                     # Z [..., 4, D] -> [..., 4, 2]
        out = np.stack([predict(probes[t][0].transform(Z[..., k, :].reshape(-1, Z.shape[-1]).astype(np.float64)),
                                probes[t][1], probes[t][2]).reshape(Z.shape[:-2] + (2,)) for k, t in enumerate(range(4, 8))], -2)
        return out

    files = sorted(FWD.glob("group_*.npz"))
    G = {}
    for f in files:
        z = np.load(f)
        for k in ["idx", "src_pred_pooled", "twin_pred_pooled", "src_real_future_pooled", "twin_real_future_pooled"] + \
                 [f"edit_pred_pooled_L{L}" for L in LAYERS]:
            G.setdefault(k, []).append(z[k])
    G = {k: np.concatenate(v) for k, v in G.items()}
    plan = np.load(FWD.parent / "plan.npz")
    idx = G["idx"]
    rows = plan["carrier_rows"][idx]
    C, T = plan["targets"][idx].shape
    src_true = cen[rows][:, 4:8]                                       # [C, 4, 2]
    twin_true = np.full((C, T, 4, 2), np.nan)
    tcache = FWD.parent / "centroids_twins.npy"
    if tcache.exists():
        twin_true = np.load(tcache)
    else:
        for i, ci in enumerate(idx):
            for j in range(T):
                twin_true[i, j] = centroids(decode(FWD.parent / "twins" / f"twin_{int(ci):04d}_{j}.mp4"))[4:8]
        np.save(tcache, twin_true)
    d = lambda a, b: np.linalg.norm(a - b, axis=-1)                   # noqa: E731
    p_src, p_twin = read(G["src_pred_pooled"]), read(G["twin_pred_pooled"])
    p_src_real, p_twin_real = read(G["src_real_future_pooled"]), read(G["twin_real_future_pooled"])
    base = d(src_true[:, None], twin_true)                            # how far the twin disk is from the source disk
    out = {"probe": "ridge per future tubelet t=4..7, real timepool[final LN, t] -> disk pixel centroid (256 px frame), "
                    "probe clips, alpha by fold CV", "real_test_clips": real_test,
           "px_units": "pixels in the 256x256 frame; a disk radius is ~10.6 px",
           "reference": {
               "src_disk_to_twin_disk_px": boot(np.nanmean(base, axis=(1, 2))),
               "probe_on_real_src_future_vs_src_true": boot(np.nanmean(d(p_src_real, src_true), 1)),
               "probe_on_real_twin_future_vs_twin_true": boot(np.nanmean(d(p_twin_real, twin_true), axis=(1, 2))),
               "predictor_src_vs_src_true": boot(np.nanmean(d(p_src, src_true), 1)),
               "predictor_twin_vs_twin_true": boot(np.nanmean(d(p_twin, twin_true), axis=(1, 2))),
               "predictor_src_vs_twin_true": boot(np.nanmean(d(p_src[:, None], twin_true), axis=(1, 2)))},
           "per_layer": {}}
    shuf = [(j + 1) % T for j in range(T)]
    for L in LAYERS:
        E = read(G[f"edit_pred_pooled_L{L}"])                          # [C, A, T, 4, 2]
        row = {}
        for a, arm in enumerate(ARMS):
            e_tw = d(E[:, a], twin_true)
            e_sh = d(E[:, a], twin_true[:, shuf])
            moved = d(E[:, a], p_src[:, None])
            # fraction of the predictor's own src->twin forecast displacement the edit recovers, projected
            v_t = p_twin - p_src[:, None]
            v_e = E[:, a] - p_src[:, None]
            rec = (v_e * v_t).sum(-1) / np.maximum((v_t * v_t).sum(-1), 1e-9)
            row[arm] = {"err_to_twin_true_px": boot(np.nanmean(e_tw, axis=(1, 2))),
                        "err_to_shuffled_twin_px": boot(np.nanmean(e_sh, axis=(1, 2))),
                        "shift_from_unedited_forecast_px": boot(np.nanmean(moved, axis=(1, 2))),
                        "recovery_of_twin_forecast_shift": boot(np.nanmean(rec, axis=(1, 2)))}
        out["per_layer"][str(L)] = row
    write_json(RES / "session2_future_position.json", out)
    return out


def timerev_speed():
    df = load_table("direction")
    d0 = load_inputs("direction", 0)
    probe, test = d0["role"] == "probe", d0["role"] == "test"
    vel = df["motion"].to_numpy() == "velocity"
    y = df["speed_mps"].to_numpy(float).reshape(-1, 1)
    F = np.load(ACT / "vjepa2" / "meanpool.npy", mmap_mode="r")
    R = np.load(ACT / "vjepa2_timerev" / "meanpool.npy", mmap_mode="r")
    ids_tr = json.loads((ACT / "vjepa2_timerev" / "ids.json").read_text())
    pos = {int(i): k for k, i in enumerate(ids_tr)}
    sel = test & vel & np.isin(df["id"].to_numpy(), ids_tr)
    rr = np.array([pos[int(i)] for i in df["id"].to_numpy()[sel]])
    m = probe & vel
    folds = d0["fold"][m]
    rows = []
    for p in range(26):
        X = np.asarray(F[:, p], np.float64)
        st = Standardizer().fit(X[m])
        cv = cv_select_alpha(st.transform(X[m]), y[m], folds, score_fn=lambda a, b: score(a, b, "linear"))
        W, b = fit_ridge(st.transform(X[m]), y[m], cv["alpha"])
        pf = predict(st.transform(X[sel]), W, b)
        pr = predict(st.transform(np.asarray(R[rr, p], np.float64)), W, b)
        sf, sr = score(y[sel], pf, "linear"), score(y[sel], pr, "linear")
        rows.append({"point": p, "r2_forward": sf["r2"], "r2_reversed": sr["r2"], "mae_forward": sf["mae"],
                     "mae_reversed": sr["mae"], "mean_pred_reversed_minus_forward": float(np.mean(pr - pf))})
    write_json(RES / "session2_timerev_speed.json",
               {"n_test_clips": int(sel.sum()), "clips": "velocity clips only (constant speed; reversal keeps speed)",
                "probe": "ridge speed_mps on forward velocity probe clips per point, alpha by fold CV", "rows": rows})
    return rows


if __name__ == "__main__":
    what = sys.argv[1:] or ["position", "speed"]
    if "speed" in what:
        r = timerev_speed()
        print("timerev speed (point 25):", r[25])
    if "position" in what:
        o = future_position()
        print(json.dumps(o["reference"], indent=1))

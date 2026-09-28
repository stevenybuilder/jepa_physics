"""Is direction's ring a dense, occupied manifold at the level of real clips, or only a ring of centroids?
(PART2_SECOND_LOOK.md section D.) CPU only, stored meanpool activations; roles exactly as run_part2 (knot folds 0-2
build, probe folds 3-4 are the real-clip reference, test clips are carriers and the floor).

A. Occupancy of the ring plane (no steering). The ring plane is the supervised circular chart X ~ mu + A[cos, sin]
   fit on knot clips (geometry_checks.fit_circular_chart); held-out clips (probe + test folds) are placed in chart
   coordinates c = pinv(A)(x - mu), where a clip on the ring has |c| = 1. Reported: the radius distribution of
   held-out clips (is there a hole in the middle?), the largest angular gap and the per-bin counts (is the ring
   occupied along its whole length?), the pooled within-value SD in chart units against the spacing of neighbouring
   values (do neighbouring values' clouds overlap into a continuum?), and the radius of chord midpoints between
   knot centroids at 90/135/180 degrees with the fraction of real clips inside that radius. The speed set (0.25-4
   m/s) is projected with the same chart to see whether slow clips fill the interior (the cone).

B. Density along the contiguous-arc steering paths (run_part2.build / pick_clips / subspace_arms, spline=smooth,
   K = 49 so waypoint 24 is the midpoint). Per waypoint: mean distance to the 5 nearest probe-fold clips in three
   spaces (full activation, PCA-64, the 2-D chart plane), divided by the median of the same statistic for unsteered
   test clips (1 = as close to real data as a real clip). Also the fraction of interior waypoints beyond the 95th
   percentile of the test clips' own 5-NN distance ("off-support rate"), and a kNN label readout in PCA-64 (circular
   mean of the 10 nearest probe clips' labels and its resultant length) at the midpoint, scored against the path's
   nominal midpoint direction on the spline's traversed arc. None of these uses the spline or a probe; they ask
   whether the steered state sits among real clips, and among real clips of the intermediate direction.
   Arms: manifold (spline, rigid shift of the clip's offset), manifold_transport (offset rotated with the loop),
   linear (chord), reflected. Paired gaps vs linear with a clip-grouped bootstrap.

  python scripts/run_ring_occupancy.py --layers 12 22 --seeds 0 1 2 ... 16
  --labels-angle-seeds 4 8 9 10: those seeds build on the labels angle (run_part2 --labels-angle), as the point-22
  all-labels arc set does; every other seed keeps run_part2's automatic angle choice.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from wm import geometry_checks as gc
from wm import manifold as mf
from wm.data import PROJECT_ROOT
from wm.p2_data import load_inputs
from wm.provenance import layer_role, provenance

ARMS = ("manifold", "manifold_transport", "linear", "reflected")
SPACES = ("full", "pca64", "chart2")
BINS = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 180.01)]
K = 49
MID = K // 2


def load_run_part2():
    spec = importlib.util.spec_from_file_location("run_part2", PROJECT_ROOT / "scripts" / "run_part2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def chart_coords(chart, X):
    return (np.asarray(X, float) - chart["mu"]) @ np.linalg.pinv(chart["A"]).T


def knn_dist(A, B, k):
    return mf.knn_distance(A, B, k)


def knn_labels(A, B, labels, k=10, chunk=2048):
    """Circular mean (deg) and resultant length of the labels of the k nearest rows of B, for each row of A."""
    A = np.asarray(A, float).reshape(-1, A.shape[-1])
    B = np.asarray(B, float)
    b2 = (B ** 2).sum(1)
    ang, rl = np.empty(len(A)), np.empty(len(A))
    u = np.exp(1j * np.radians(labels))
    for i in range(0, len(A), chunk):
        a = A[i:i + chunk]
        d2 = (a ** 2).sum(1)[:, None] + b2[None] - 2 * a @ B.T
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        z = u[idx].mean(1)
        ang[i:i + chunk], rl[i:i + chunk] = np.degrees(np.angle(z)) % 360.0, np.abs(z)
    return ang, rl


def occupancy(d, chart, speed_set=None):
    knot, held = d["role"] == "knot", d["role"] != "knot"
    y = d["y"]
    c = chart_coords(chart, d["X"][held])
    r, a = np.linalg.norm(c, axis=1), np.degrees(np.arctan2(c[:, 1], c[:, 0])) % 360.0
    sa = np.sort(a)
    gaps = np.diff(np.append(sa, sa[0] + 360.0))
    counts = np.histogram(a, bins=64, range=(0, 360))[0]
    # pooled within-value SD in chart units (knot clips)
    ck = chart_coords(chart, d["X"][knot])
    res = ck.copy()
    for v in np.unique(y[knot]):
        rows = y[knot] == v
        res[rows] -= ck[rows].mean(0)
    wsd = float(np.sqrt((res ** 2).mean()))
    # chord midpoints between knot centroids, in chart units
    vals = np.unique(y[knot])
    C = np.array([ck[y[knot] == v].mean(0) for v in vals])
    mids = {}
    for dth in (45.0, 90.0, 135.0, 180.0):
        rr = []
        for i, v in enumerate(vals):
            j = int(np.argmin(np.abs(((vals - v - dth) + 180) % 360 - 180)))
            rr.append(np.linalg.norm((C[i] + C[j]) / 2))
        rm = float(np.median(rr))
        mids[f"{dth:.0f}"] = {"chord_midpoint_radius": rm, "frac_heldout_clips_inside": float(np.mean(r < rm))}
    out = {"n_heldout_clips": int(held.sum()),
           "radius_quantiles": {q: float(np.quantile(r, float(q))) for q in ("0.01", "0.05", "0.25", "0.5", "0.75")},
           "frac_radius_below": {t: float(np.mean(r < float(t))) for t in ("0.25", "0.5", "0.75")},
           "centroid_radius_median": float(np.median(np.linalg.norm(C, axis=1))),
           "max_angular_gap_deg": float(gaps.max()), "clips_per_5.625deg_bin_min_max": [int(counts.min()),
                                                                                         int(counts.max())],
           "within_value_sd_chart_units": wsd, "neighbour_spacing_chart_units": float(2 * np.pi / 64),
           "overlap_ratio_sd_over_spacing": wsd / float(2 * np.pi / 64),
           "chord_midpoints": mids,
           "note": "chart units: a clip on the fitted ring has radius 1; the ring's own radius in PCA units is "
                   "p2_geometry_direction_L*.json circular_chart.chart_radius"}
    if "speed_mps" in d["df"]:
        sp = d["df"]["speed_mps"].to_numpy(float)[held]
        out["radius_by_speed_tercile"] = [{"speed_range": [float(sp[m].min()), float(sp[m].max())],
                                           "median_radius": float(np.median(r[m]))}
                                          for m in np.array_split(np.argsort(sp), 3)]
    if speed_set is not None:
        cs = chart_coords(chart, speed_set["X"])
        rs = np.linalg.norm(cs, axis=1)
        sp = speed_set["df"]["speed_mps"].to_numpy(float)
        th = speed_set["df"]["theta_degrees"].to_numpy(float)
        ang = np.degrees(np.arctan2(cs[:, 1], cs[:, 0])) % 360.0
        rows = []
        for idx in np.array_split(np.argsort(sp), 8):
            e = np.abs(((ang[idx] - th[idx]) + 180) % 360 - 180)
            rows.append({"speed_range": [float(sp[idx].min()), float(sp[idx].max())],
                         "median_radius": float(np.median(rs[idx])), "frac_radius_below_0.5": float(np.mean(rs[idx] < 0.5)),
                         "chart_angle_mae_deg": float(e.mean())})
        out["speed_set_in_direction_chart"] = {
            "by_speed_octile": rows,
            "note": "speed-set clips (0.25-4 m/s, all 64 directions) placed with the direction-set chart; the chart "
                    "angle is scored against theta_degrees (orientation fixed by the fit)"}
    return out


def boot_ci(diff, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    means = []
    for _ in range(n):
        s = rng.choice(ug, len(ug))
        means.append(np.concatenate([diff[idx[g]] for g in s]).mean())
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paths(rp2, d, seed, chart_all_knot=None, angle="unsupervised"):
    m = rp2.build(d, 64, angle, "contiguous", seed, n_controls=0, spline="smooth")
    knot, probe, test = m["knot"], d["role"] == "probe", d["role"] == "test"
    chart = gc.fit_circular_chart(d["X"][knot], d["y"][knot])      # kept values only, like the spline
    ref = {"full": d["X"][probe].astype(float), "pca64": m["pca"].project(d["X"][probe]),
           "chart2": chart_coords(chart, d["X"][probe])}
    to_space = {"full": lambda W: W, "pca64": lambda W: m["pca"].project(W), "chart2": lambda W: chart_coords(chart, W)}
    floor = {}
    for s in SPACES:
        f = knn_dist(to_space[s](d["X"][test].astype(float)), ref[s], 5)
        floor[s] = {"median": float(np.median(f)), "p95": float(np.percentile(f, 95))}
    ylab = d["y"][probe]
    picks = rp2.pick_clips(d, m["held"], 48, seed)
    rec = {a: {q: [] for q in ("mid_ratio", "max_ratio", "excess_ratio", "offsupport")} for a in ARMS}
    lab = {a: {"mid_err": [], "mid_rbar": []} for a in ARMS}
    shift_all, clip_all = [], []
    for tgt, pick in picks.items():
        x, src = d["X"][pick].astype(float), d["y"][pick]
        Z, resid = m["pca"].project(x), m["pca"].complement(x)
        Zarms = rp2.subspace_arms(Z, src, tgt, m, K)
        shift_all.append(mf.value_error(src, tgt, True))
        clip_all.append(pick)
        sign = rp2.traversed_arc(m["curve"], src, tgt)
        span = ((tgt - src) * sign) % 360.0
        nominal_mid = (src + sign * span / 2) % 360.0
        for a in ARMS:
            W = rp2.compose(m["pca"], Zarms[a], resid)                  # [n, K, D]
            n = len(W)
            ratios = {}
            for s in SPACES:
                dist = knn_dist(to_space[s](W.reshape(n * K, -1)), ref[s], 5).reshape(n, K)
                ratios[s] = dist / floor[s]["median"]
            rec[a]["mid_ratio"].append(np.stack([ratios[s][:, MID] for s in SPACES], 1))
            rec[a]["max_ratio"].append(np.stack([ratios[s][:, 1:-1].max(1) for s in SPACES], 1))
            rec[a]["excess_ratio"].append(np.stack([ratios[s].mean(1) - ratios[s][:, 0] for s in SPACES], 1))
            rec[a]["offsupport"].append(np.stack(
                [(ratios[s][:, 1:-1] * floor[s]["median"] > floor[s]["p95"]).mean(1) for s in SPACES], 1))
            ang, rbar = knn_labels(to_space["pca64"](W[:, MID]), ref["pca64"], ylab, k=10)
            lab[a]["mid_err"].append(mf.value_error(ang, nominal_mid, True))
            lab[a]["mid_rbar"].append(rbar)
    shift = np.concatenate(shift_all)
    clips = np.concatenate(clip_all)
    R = {a: {q: np.concatenate(v) for q, v in rec[a].items()} for a in ARMS}
    L = {a: {q: np.concatenate(v) for q, v in lab[a].items()} for a in ARMS}
    return {"m": m, "floor": floor, "shift": shift, "clips": clips, "R": R, "L": L, "held": m["held"]}


def summarise_paths(P, seed=0):
    shift, clips, R, L = P["shift"], P["clips"], P["R"], P["L"]
    out = {"floor_5nn": P["floor"], "held_out_arc": [float(P["held"][0]), float(P["held"][-1])], "by_shift": [],
           "overall": {}, "gaps": {}}
    for lo, hi in [(0, 180.01)] + BINS:
        sel = (shift >= lo) & (shift < hi)
        row = {"shift_lo": lo, "shift_hi": hi, "n": int(sel.sum())}
        for a in ARMS:
            row[a] = {f"{q}_{s}": float(R[a][q][sel, i].mean()) for q in R[a] for i, s in enumerate(SPACES)}
            row[a]["knn_mid_err_deg"] = float(L[a]["mid_err"][sel].mean())
            row[a]["knn_mid_resultant"] = float(L[a]["mid_rbar"][sel].mean())
        if lo == 0 and hi > 180:
            out["overall"] = row
        else:
            out["by_shift"].append(row)
    for a in ("manifold", "manifold_transport", "reflected"):
        g = {}
        for q in ("mid_ratio", "max_ratio", "excess_ratio", "offsupport"):
            for i, s in enumerate(SPACES):
                diff = R[a][q][:, i] - R["linear"][q][:, i]
                g[f"{q}_{s}"] = {"mean": float(diff.mean()), "ci95": boot_ci(diff, clips, seed=seed),
                                 "large_shift_120_180": float(diff[shift >= 120].mean())}
        for q in ("mid_err", "mid_rbar"):
            diff = L[a][q] - L["linear"][q]
            g[f"knn_{q}"] = {"mean": float(diff.mean()), "ci95": boot_ci(diff, clips, seed=seed),
                             "large_shift_120_180": float(diff[shift >= 120].mean())}
        out["gaps"][f"{a}_minus_linear"] = g
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, nargs="+", default=[12, 22])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--out", default=str(PROJECT_ROOT / "results" / "p2_ring_occupancy.json"))
    p.add_argument("--labels-angle-seeds", type=int, nargs="*", default=[],
                   help="seeds built on the labels angle (run_part2 --labels-angle) instead of the automatic choice")
    args = p.parse_args(argv)
    rp2 = load_run_part2()
    res = {"provenance": provenance(None, seeds={"contiguous_arc_seeds": args.seeds,
                                                 "labels_angle_seeds": args.labels_angle_seeds}, layers=args.layers,
                                    pool="meanpool", k=64, K=K, spline="smooth", holdout="contiguous"),
           "question": "is the ring occupied by real clips along its whole length with an empty interior, and do "
                       "steered waypoints sit among real clips (of the intermediate direction)?",
           "metric_notes": {
               "ratio": "mean distance to the 5 nearest probe-fold clips / median of the same for unsteered test "
                        "clips, in that space (1 = as close to real data as a real clip)",
               "mid": "waypoint 24 of 49 (the path midpoint)", "max": "maximum over interior waypoints",
               "excess_ratio": "mean ratio along the path minus the unsteered clip's own ratio (waypoint 0)",
               "offsupport": "fraction of interior waypoints whose 5-NN distance exceeds the 95th percentile of the "
                             "test clips' own 5-NN distance",
               "knn_mid_err_deg": "circular-mean label of the 10 nearest probe clips (PCA-64) at the midpoint vs the "
                                  "nominal midpoint direction on the spline's traversed arc",
               "knn_mid_resultant": "resultant length of those 10 labels (1 = all the same direction, ~0.3 = "
                                    "scattered)",
               "spaces": "full = 1024-d meanpool; pca64 = run_part2's PCA-64 (knot clips, kept values); chart2 = "
                         "the circular-chart plane in chart units"},
           "layers": {}}
    for L in args.layers:
        d = load_inputs("direction", L)
        s = load_inputs("speed", L)
        chart = gc.fit_circular_chart(d["X"][d["role"] == "knot"], d["y"][d["role"] == "knot"])
        entry = {**layer_role("direction", L, "direction"), "occupancy": occupancy(d, chart, s), "paths": {}}
        for seed in args.seeds:
            P = paths(rp2, d, seed, angle="labels" if seed in args.labels_angle_seeds else "unsupervised")
            entry["paths"][f"s{seed}"] = summarise_paths(P, seed)
            entry["paths"][f"s{seed}"]["angle_source"] = P["m"]["curve"].coord_source
            o = entry["paths"][f"s{seed}"]["overall"]
            print(f"L{L} s{seed}: mid ratio pca64 spline {o['manifold']['mid_ratio_pca64']:.3f} chord "
                  f"{o['linear']['mid_ratio_pca64']:.3f} | chart2 {o['manifold']['mid_ratio_chart2']:.2f} vs "
                  f"{o['linear']['mid_ratio_chart2']:.2f}", flush=True)
        if len(args.seeds) > 1:
            agg = {}
            for a in ("manifold", "manifold_transport", "reflected"):
                for q in ("mid_ratio_full", "mid_ratio_pca64", "mid_ratio_chart2", "excess_ratio_full",
                          "excess_ratio_pca64", "offsupport_pca64", "knn_mid_err", "knn_mid_rbar"):
                    v = np.array([entry["paths"][f"s{sd}"]["gaps"][f"{a}_minus_linear"][q]["mean"] for sd in args.seeds])
                    vl = np.array([entry["paths"][f"s{sd}"]["gaps"][f"{a}_minus_linear"][q]["large_shift_120_180"]
                                   for sd in args.seeds])
                    ci = np.array([entry["paths"][f"s{sd}"]["gaps"][f"{a}_minus_linear"][q]["ci95"] for sd in args.seeds])
                    agg[f"{a}_minus_linear.{q}"] = {"mean_over_arcs": float(v.mean()), "sd_over_arcs": float(v.std(ddof=1)),
                                                    "large_shift_mean_over_arcs": float(vl.mean()),
                                                    "n_ci_above_0": int((ci[:, 0] > 0).sum()),
                                                    "n_ci_below_0": int((ci[:, 1] < 0).sum()), "n_arcs": len(v)}
            entry["over_arcs"] = agg
        res["layers"][str(L)] = entry
        Path(args.out).write_text(json.dumps(res, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

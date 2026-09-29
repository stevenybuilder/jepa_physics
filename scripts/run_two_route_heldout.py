"""Held-out two-route 180-degree test (the Part 2 second-look note, B.5; literature review item 3(b), the long way round).

scripts/run_two_route.py is exploratory: its spline is built on all 64 values. Here every run uses run_part2's
contiguous held-out design (run_part2.build(..., "contiguous", seed, spline="smooth"): one 45-degree arc of 8 values
left out of PCA-64, the centroids and the spline, the arc chosen by the seed, exactly as results/arcs_rawchord and
p2_steer_direction_direction_L*_contiguous_rawchord.json). Readouts are fit on disjoint clips (linear probe and MLP
on probe folds 3-4, Eq. 9 behaviour manifold on the same probe folds); carriers are test clips.

Two pair designs per held value h (both run for every seed):
  antipode_in_arc   source h - 180, target h (the antipode is held out; the routes enter the arc from either side)
  midpoint_in_arc   source h - 90, target h + 90 (the via-+90 route crosses the held arc at its midpoint h; the
                    via--90 route crosses kept values only)
Routes (K = 50 waypoints, PCA-64 shift mode, the clip's off-subspace residual kept, as run_two_route.paths):
  via_plus90    along the held-out spline through source + 90 (increasing label angle)
  via_minus90   along the held-out spline through source - 90
  chord         straight line between the smoothed-knot polyline points at source and target (run_two_route's chord)
  chord_raw     straight line between the raw-centroid polyline points (run_part2's linear_raw)
The two spline routes end at the same point by construction (shift mode on a closed spline), so their endpoint errors
are equal; the comparison there is spline vs chord.

Per carrier and route: endpoint probe / MLP error to the target; minimum and midpoint linear-probe (sin, cos) radius
(start radius for scale); ordering (Spearman between waypoint index and the readout's offset along the route's arc,
path_value_stats on a one-hot of the readout snapped to the 64-value grid); intermediate mass = mean over interior
waypoints of the mass strictly between source and target on the route's half-ring (probe one-hot, MLP one-hot and
Eq. 9); midpoint probe error to the route's nominal midpoint (source +- 90). Chord: the same, scored on each half.
Aggregation: per-seed (arc) means, then mean over the 16 arcs with a 95% bootstrap CI over arcs.

  python scripts/run_two_route_heldout.py --layers 12 22 --seeds 1 2 ... 16
"""
import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))  # run without installing wm
from wm import manifold as mf
from wm.bakeoff import MLPReadout
from wm.data import PROJECT_ROOT
from wm.p2_data import load_inputs
from wm.provenance import layer_role, provenance

ROUTES = ("via_plus90", "via_minus90", "chord", "chord_raw")
DESIGNS = ("antipode_in_arc", "midpoint_in_arc")
ROUTE_SIGN = {"via_plus90": 1.0, "via_minus90": -1.0}
LABELS_ANGLE_SEEDS = {22: (4, 8, 9, 10)}   # the point-22 all-labels arc set (run_ring_occupancy / arcs_rawchord)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def one_hot(angles, values):
    """[..] angles (deg) -> [.., V] one-hot on the nearest grid value."""
    step = 360.0 / len(values)
    idx = np.rint(((np.asarray(angles) - values[0]) % 360.0) / step).astype(int) % len(values)
    return np.eye(len(values))[idx]


def pair_list(held, design):
    held = np.asarray(held, float)
    if design == "antipode_in_arc":
        return [((h - 180.0) % 360.0, h) for h in held]
    return [((h - 90.0) % 360.0, (h + 90.0) % 360.0) for h in held]


def route_waypoints(rtr, m, Z, src, K):
    P = rtr.paths(Z, m["curve"], src, K)                                   # via_plus90, via_minus90, chord
    tgt = (src + 180.0) % 360.0
    P["chord_raw"] = mf.linear_coords(Z, mf.piecewise_linear_point(m["raw_curve"], src),
                                      mf.piecewise_linear_point(m["raw_curve"], tgt), K)
    return P


def score_route(W, src, tgt, r, ev, values, bm):
    """Per-carrier metrics of waypoints W [n, K, D] on route r."""
    n, K = W.shape[:2]
    flat = W.reshape(n * K, -1)
    raw = ev["probe"].raw(flat).reshape(n, K, 2)
    pang = np.degrees(np.arctan2(raw[..., 0], raw[..., 1])) % 360.0
    prad = np.linalg.norm(raw, axis=-1)
    mang = ev["mlp"].predict(flat).reshape(n, K)
    Pb = bm.value_distribution(W, bm.F(W))
    mid = K // 2
    out = {"end_probe_err": mf.value_error(pang[:, -1], tgt, True),
           "end_mlp_err": mf.value_error(mang[:, -1], tgt, True),
           "probe_radius_start": prad[:, 0], "probe_radius_min": prad.min(1), "probe_radius_mid": prad[:, mid],
           "probe_radius_min_over_start": prad.min(1) / prad[:, 0]}
    signs = {"": ROUTE_SIGN[r]} if r in ROUTE_SIGN else {"_plus_half": 1.0, "_minus_half": -1.0}
    for suf, sg in signs.items():
        arc = np.full(n, sg)
        for name, Pv in (("probe", one_hot(pang, values)), ("mlp", one_hot(mang, values)), ("eq9", Pb)):
            st = mf.path_value_stats(Pv, src, tgt, values, True, arc)
            out[f"{name}_intermediate_mass{suf}"] = st["intermediate"][:, 1:-1].mean(1)
            if name != "eq9":
                out[f"{name}_ordering{suf}"] = st["ordering"]
        out[f"probe_mid_err_to_route_mid{suf}"] = mf.value_error(pang[:, mid], (src + sg * 90.0) % 360.0, True)
        out[f"mlp_mid_err_to_route_mid{suf}"] = mf.value_error(mang[:, mid], (src + sg * 90.0) % 360.0, True)
    return out, {"probe_offset": signed(pang, src), "mlp_offset": signed(mang, src), "probe_radius": prad}


def signed(a, src):
    return (a - src[:, None] + 180.0) % 360.0 - 180.0


def boot_mean_ci(v, n=4000, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return {"mean": float(v.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "n_arcs": int(len(v))}


def run_seed(rp2, rtr, d, seed, angle, ev, K):
    m = rp2.build(d, 64, angle, "contiguous", seed, n_controls=0, spline="smooth")
    bm = m["behaviour"]
    values = np.asarray(bm.values, float)
    test = np.flatnonzero(d["role"] == "test")
    res, traces = {}, {}
    for design in DESIGNS:
        acc = {r: {} for r in ROUTES}
        tr = {r: [] for r in ROUTES}
        n_car = 0
        for src_v, tgt_v in pair_list(m["held"], design):
            rows = test[np.isclose(d["y"][test], src_v)]
            x = d["X"][rows].astype(float)
            src = d["y"][rows]
            tgt = np.full(len(rows), tgt_v)
            Z, resid = m["pca"].project(x), m["pca"].complement(x)
            P = route_waypoints(rtr, m, Z, src, K)
            n_car += len(rows)
            for r in ROUTES:
                W = rp2.compose(m["pca"], P[r], resid)
                met, t = score_route(W, src, tgt, r, ev, values, bm)
                for q, v in met.items():
                    acc[r].setdefault(q, []).append(v)
                tr[r].append(t)
        per = {r: {q: float(np.mean(np.concatenate(v))) for q, v in acc[r].items()} for r in ROUTES}
        res[design] = {"n_carriers": n_car, "routes": per}
        traces[design] = {r: {k: np.concatenate([t[k] for t in tr[r]]) for k in tr[r][0]} for r in ROUTES}
    info = {"held_out_arc": [float(m["held"][0]), float(m["held"][-1])], "angle_source": m["curve"].coord_source,
            "block_start_index": m["design"]["block_start_index"]}
    return res, traces, info


def aggregate(per_seed):
    out = {}
    for design in DESIGNS:
        agg = {}
        for r in ROUTES:
            keys = per_seed[0][design]["routes"][r].keys()
            agg[r] = {q: boot_mean_ci([s[design]["routes"][r][q] for s in per_seed]) for q in keys}
        # paired contrasts over arcs
        def v(r, q):
            return np.array([s[design]["routes"][r][q] for s in per_seed])
        c = {}
        for sp in ("via_plus90", "via_minus90"):
            for ch in ("chord", "chord_raw"):
                d_ = v(sp, "probe_radius_min") - v(ch, "probe_radius_min")
                c[f"{sp}_minus_{ch}.probe_radius_min"] = {**boot_mean_ci(d_), "n_arcs_positive": int((d_ > 0).sum())}
                d_ = v(sp, "end_probe_err") - v(ch, "end_probe_err")
                c[f"{sp}_minus_{ch}.end_probe_err"] = {**boot_mean_ci(d_), "n_arcs_positive": int((d_ > 0).sum())}
        for q in ("probe_intermediate_mass", "mlp_intermediate_mass", "eq9_intermediate_mass"):
            # each spline route's mass on its own half vs the chord's mass on that same half
            for sp, half in (("via_plus90", "_plus_half"), ("via_minus90", "_minus_half")):
                d_ = v(sp, q) - v("chord", q + half)
                c[f"{sp}_minus_chord.{q}_same_half"] = {**boot_mean_ci(d_), "n_arcs_positive": int((d_ > 0).sum())}
        agg["contrasts"] = c
        out[design] = agg
    return out


def verdict(L, agg):
    a = agg["antipode_in_arc"]
    b = agg["midpoint_in_arc"]

    def f(dct, r, q):
        e = dct[r][q]
        return f"{e['mean']:.2f} [{e['ci95'][0]:.2f}, {e['ci95'][1]:.2f}]"
    return (f"point {L} (16 held-out 45-degree arcs, antipode in the arc): probe intermediate mass on the route's "
            f"half-ring via +90 {f(a, 'via_plus90', 'probe_intermediate_mass')}, via -90 "
            f"{f(a, 'via_minus90', 'probe_intermediate_mass')} (chord on + half "
            f"{f(a, 'chord', 'probe_intermediate_mass_plus_half')}, - half "
            f"{f(a, 'chord', 'probe_intermediate_mass_minus_half')}); ordering (probe Spearman) "
            f"{f(a, 'via_plus90', 'probe_ordering')} / {f(a, 'via_minus90', 'probe_ordering')}; min probe radius "
            f"spline {f(a, 'via_plus90', 'probe_radius_min')} / {f(a, 'via_minus90', 'probe_radius_min')} vs chord "
            f"{f(a, 'chord', 'probe_radius_min')} (raw chord {f(a, 'chord_raw', 'probe_radius_min')}; start "
            f"{f(a, 'chord', 'probe_radius_start')}); endpoint probe error spline {f(a, 'via_plus90', 'end_probe_err')}"
            f" deg vs chord {f(a, 'chord', 'end_probe_err')} (raw {f(a, 'chord_raw', 'end_probe_err')}). Midpoint in "
            f"the arc: via +90 (crosses the held arc) intermediate mass {f(b, 'via_plus90', 'probe_intermediate_mass')}"
            f", midpoint probe error to the held value {f(b, 'via_plus90', 'probe_mid_err_to_route_mid')} deg; via -90 "
            f"{f(b, 'via_minus90', 'probe_intermediate_mass')}, {f(b, 'via_minus90', 'probe_mid_err_to_route_mid')} "
            f"deg.")


def waypoint_summary(tr):
    """Per-waypoint curves pooled over carriers and arcs: circular-mean probe offset and radius mean / 10-90%."""
    out = {}
    for ds in DESIGNS:
        out[ds] = {}
        for r in ROUTES:
            c = tr[ds][r]
            rad = c["probe_radius"]
            out[ds][r] = {"probe_offset_circmean": np.degrees(np.angle(np.exp(1j * np.radians(c["probe_offset"]))
                                                                       .mean(0))).tolist(),
                          "mlp_offset_circmean": np.degrees(np.angle(np.exp(1j * np.radians(c["mlp_offset"]))
                                                                     .mean(0))).tolist(),
                          "probe_radius_mean": rad.mean(0).tolist(), "probe_radius_p10": np.percentile(rad, 10, 0).tolist(),
                          "probe_radius_p90": np.percentile(rad, 90, 0).tolist(), "n": int(len(rad))}
    return out


def plot(curves, path):
    layers = sorted(curves)
    col = {"via_plus90": "#2a6fdb", "via_minus90": "#e07b00", "chord": "#555555", "chord_raw": "#aaaaaa"}
    fig, axes = plt.subplots(len(layers), 3, figsize=(12.5, 3.3 * len(layers)), squeeze=False)
    for i, L in enumerate(layers):
        for r in ROUTES:
            for j, ds in ((0, "antipode_in_arc"), (2, "midpoint_in_arc")):
                off = np.asarray(curves[L][ds][r]["probe_offset_circmean"])
                s = np.linspace(0, 1, len(off))
                if not r.startswith("chord"):
                    off = np.degrees(np.unwrap(np.radians(off)))
                axes[i, j].plot(s, off, color=col[r], label=r)
            c = curves[L]["antipode_in_arc"][r]
            axes[i, 1].plot(s, c["probe_radius_mean"], color=col[r])
            axes[i, 1].fill_between(s, c["probe_radius_p10"], c["probe_radius_p90"], color=col[r], alpha=0.12)
        axes[i, 0].set_ylabel(f"point {L}")
        axes[i, 0].set_title("antipode in held arc: probe angle offset from source (deg)", fontsize=8)
        axes[i, 1].set_title("antipode in held arc: probe (sin, cos) radius (mean, 10-90%)", fontsize=8)
        axes[i, 2].set_title("midpoint in held arc: probe angle offset (via +90 crosses the arc)", fontsize=8)
        for ax in axes[i]:
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlabel("path fraction s", fontsize=8)
        for ax in (axes[i, 0], axes[i, 2]):
            for yv in (-180, -90, 0, 90, 180):
                ax.axhline(yv, color="#dddddd", lw=0.6, zorder=0)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Two-route 180-degree test, spline fit without the held-out 45-degree arc (16 arcs, carriers pooled)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(args):
    rp2, rtr = _load("run_part2"), _load("run_two_route")
    rdir, fdir = Path(args.results_dir), Path(args.figures_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    curves = {}
    for L in ([] if args.plot_only else args.layers):
        d = load_inputs("direction", L, "direction", args.act_dir, args.table, args.split)
        probe_rows = d["role"] == "probe"
        ev = {"probe": mf.ProbeReadout(d["X"][probe_rows], d["y"][probe_rows], True),
              "mlp": MLPReadout(d["X"][probe_rows], d["y"][probe_rows], True, args.mlp_seed)}
        lab = LABELS_ANGLE_SEEDS.get(L, ()) if args.labels_angle_seeds is None else args.labels_angle_seeds
        per_seed, info, tr_all = [], {}, None
        for seed in args.seeds:
            angle = "labels" if seed in lab else "unsupervised"
            res, tr, inf = run_seed(rp2, rtr, d, seed, angle, ev, args.K)
            per_seed.append(res)
            info[f"s{seed}"] = {**inf, **{f"{ds}_n_carriers": res[ds]["n_carriers"] for ds in DESIGNS}}
            if tr_all is None:
                tr_all = tr
            else:
                for ds in DESIGNS:
                    for r in ROUTES:
                        for k in tr_all[ds][r]:
                            tr_all[ds][r][k] = np.concatenate([tr_all[ds][r][k], tr[ds][r][k]])
            print(f"L{L} s{seed} arc {inf['held_out_arc']} {inf['angle_source']}: via+90 mass "
                  f"{res['antipode_in_arc']['routes']['via_plus90']['probe_intermediate_mass']:.2f} via-90 "
                  f"{res['antipode_in_arc']['routes']['via_minus90']['probe_intermediate_mass']:.2f} rmin chord "
                  f"{res['antipode_in_arc']['routes']['chord']['probe_radius_min']:.2f}", flush=True)
        agg = aggregate(per_seed)
        out = {"dataset": "direction", "variable": "direction", "layer": L,
               "layer_role": layer_role("direction", L)["layer_role"],
               "label": "held-out: spline, PCA-64 and centroids fit without the 45-degree arc; readouts on disjoint "
                        "probe-fold clips; carriers = test clips",
               "provenance": provenance(args.split, seeds={"contiguous_arc_seeds": args.seeds, "mlp": args.mlp_seed,
                                                           "labels_angle_seeds": list(lab)},
                                        layers=[L], k=64, K=args.K, spline="smooth", holdout="contiguous"),
               "design": {"antipode_in_arc": "source h - 180, target h, for each of the 8 held values h",
                          "midpoint_in_arc": "source h - 90, target h + 90: the via-+90 route crosses the held arc at "
                                             "its midpoint, via -90 crosses kept values"},
               "routes": {"via_plus90": "held-out spline through source + 90", "via_minus90": "through source - 90",
                          "chord": "line between smoothed-knot polyline points (run_two_route chord)",
                          "chord_raw": "line between raw-centroid polyline points (run_part2 linear_raw)"},
               "metric_notes": {
                   "end_*_err": "wrapped error (deg) at the last waypoint vs the target; the two spline routes end at "
                                "the same point by construction",
                   "probe_radius_*": "linear-probe (sin, cos) norm; start = unsteered carrier",
                   "*_intermediate_mass": "mean over interior waypoints of the readout mass strictly between source "
                                          "and target on the route's half-ring (probe / MLP: one-hot of the readout "
                                          "snapped to the 64-value grid; eq9: Goodfire Eq. 9 distribution); chord "
                                          "scored on each half",
                   "*_ordering": "Spearman(waypoint index, readout offset along the route's arc)",
                   "*_mid_err_to_route_mid": "wrapped error at waypoint K//2 vs source + 90 (or - 90)",
                   "ci95": "bootstrap over the 16 arcs of the per-arc mean"},
               "per_arc_info": info, "per_arc": {f"s{s}": r for s, r in zip(args.seeds, per_seed)},
               "over_arcs": agg}
        out["per_waypoint"] = waypoint_summary(tr_all)
        out["verdict"] = verdict(L, agg)
        print(out["verdict"], flush=True)
        (rdir / f"p2_two_route_heldout_L{L}.json").write_text(json.dumps(out, indent=1))
        curves[L] = out["per_waypoint"]
    for L in args.plot_layers:                     # other layers' finished runs, for one combined figure
        f = rdir / f"p2_two_route_heldout_L{L}.json"
        if L not in curves and f.exists():
            curves[L] = json.loads(f.read_text())["per_waypoint"]
    plot(curves, fdir / "fig_two_route_heldout.png")


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, nargs="+", default=[12, 22])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 17)))
    p.add_argument("--labels-angle-seeds", type=int, nargs="*", default=None,
                   help="override the per-layer labels-angle seeds (default: point 22 -> 4 8 9 10)")
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--plot-layers", type=int, nargs="*", default=[12, 22])
    p.add_argument("--plot-only", action="store_true", help="redraw the figure from existing result JSONs")
    p.add_argument("--mlp-seed", type=int, default=0)
    p.add_argument("--act-dir", default=None)
    p.add_argument("--table", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--figures-dir", default=str(PROJECT_ROOT / "figures"))
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse())

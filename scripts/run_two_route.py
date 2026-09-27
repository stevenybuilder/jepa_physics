"""Two-route 180-degree test on the direction ring (JEPA_STEERING_LESSONS.md section 6; PART2_RATIONALE.md section 6).
EXPLORATORY: the spline is built on all 64 values, so no target is held out.

For source theta and target theta + 180 (the 32 antipodal pairs, run in both directions: every one of the 64 values is
a source), carriers = test clips at theta. Three paths of K = 50 waypoints in PCA-64 (shift mode: z + curve(t) -
curve(t_source), the clip's off-subspace residual kept):
  via_plus90    along the smoothing spline through theta + 90 (increasing label angle)
  via_minus90   along the smoothing spline through theta - 90 = theta + 270 (decreasing label angle)
  chord         the straight line between the curve points at theta and theta + 180
At every waypoint: linear-probe angle and (sin, cos) radius (mf.ProbeReadout), MLP angle (wm.bakeoff.MLPReadout), and
nearest-real R against the real test clips at the waypoint's nominal angle (spline routes: theta +- 180 s rounded to
the 5.625-degree grid; chord: its endpoint target, plus R against theta +- 90 at the midpoint). The carrier itself is
excluded from every reference set. Roles: knot folds 0-2 (all values) build PCA-64 + spline; probe folds 3-4 fit both
evaluators; test clips are carriers and the nearest-real reference.

  python scripts/run_two_route.py --layers 12 22
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wm import geometry_checks as gc
from wm import manifold as mf
from wm.bakeoff import MLPReadout
from wm.data import PROJECT_ROOT
from wm.p2_data import load_inputs
from wm.provenance import layer_role, provenance

ROUTES = ("via_plus90", "via_minus90", "chord")
TWO_PI = 2 * np.pi


def build_all_values(d, k, spline):
    knot = d["role"] == "knot"
    pca = gc.fit_subspace(d["X"][knot], d["y"][knot], "pca", k, None)
    cent = mf.centroids(pca.project(d["X"][knot]), d["y"][knot])
    choice = mf.choose_angle_source(cent["C"], cent["values"])
    curve = mf.fit_curve(cent, True, angle=choice["angle"], plane=choice["plane"], spline=spline)
    return pca, curve, choice


def route_steps(curve, theta):
    """Signed intrinsic-coordinate steps (plus, minus) from theta to theta + 180 through theta + 90 / theta - 90."""
    ta = curve.coord_of_value(theta)
    tb = curve.coord_of_value((theta + 180.0) % 360.0)
    tm = curve.coord_of_value((theta + 90.0) % 360.0)
    fwd = (tb - ta) % TWO_PI                       # forward (increasing intrinsic) step, in (0, 2 pi)
    mid_fwd = (tm - ta) % TWO_PI
    plus = np.where(mid_fwd < fwd, fwd, fwd - TWO_PI)
    minus = np.where(mid_fwd < fwd, fwd - TWO_PI, fwd)
    return ta, plus, minus


def paths(Z, curve, theta, K):
    """PCA-coordinate waypoints [n, K, k] of each route for clips at PCA coordinates Z [n, k], source theta [n]."""
    s = np.linspace(0.0, 1.0, K)
    ta, plus, minus = route_steps(curve, theta)
    out = {}
    for name, step in (("via_plus90", plus), ("via_minus90", minus)):
        t = ta[:, None] + s[None] * step[:, None]
        on = curve(t)
        out[name] = Z[:, None] + on - on[:, :1]
    pa = mf.piecewise_linear_point(curve, theta)
    pb = mf.piecewise_linear_point(curve, (theta + 180.0) % 360.0)
    out["chord"] = mf.linear_coords(Z, pa, pb, K)
    return out


def circ_sd_deg(a):
    """Circular standard deviation (degrees) of angles a (degrees)."""
    Rbar = np.abs(np.exp(1j * np.radians(a)).mean())
    return float(np.degrees(np.sqrt(-2 * np.log(max(Rbar, 1e-12)))))


def signed_offset(a, ref):
    return (np.asarray(a) - np.asarray(ref) + 180.0) % 360.0 - 180.0


class RefMeans:
    """Mean of the real test clips at each grid value, leaving out one given row."""

    def __init__(self, X, y, rows):
        self.X, self.y, self.rows = X, y, rows
        self.values = np.unique(y)
        self.sum = {v: X[rows[np.isclose(y[rows], v)]].astype(float).sum(0) for v in self.values}
        self.cnt = {v: int(np.isclose(y[rows], v).sum()) for v in self.values}

    def snap(self, a):
        step = 360.0 / len(self.values)
        return self.values[np.round((np.asarray(a) % 360.0) / step).astype(int) % len(self.values)]

    def mean(self, v, exclude):
        s, c = self.sum[v], self.cnt[v]
        if np.isclose(self.y[exclude], v):
            s, c = s - self.X[exclude], c - 1
        return s / c


def run_layer(d, L, args):
    pca, curve, choice = build_all_values(d, args.k, args.spline)
    probe_rows = d["role"] == "probe"
    probe = mf.ProbeReadout(d["X"][probe_rows], d["y"][probe_rows], True)
    mlp = MLPReadout(d["X"][probe_rows], d["y"][probe_rows], True, args.seed)
    test = np.flatnonzero(d["role"] == "test")
    ref = RefMeans(d["X"], d["y"], test)
    X, y = d["X"][test].astype(float), d["y"][test]
    Z, resid = pca.project(X), pca.complement(X)
    K = args.K
    s = np.linspace(0, 1, K)
    P = paths(Z, curve, y, K)
    mid = K // 2
    out = {"angle_choice": {k: v for k, v in choice.items() if k != "checks"}, "n_carriers": int(len(test)),
           "n_source_values": int(len(np.unique(y))), "routes": {}}
    curves = {}
    for r in ROUTES:
        W = pca.lift(P[r]) + resid[:, None]                          # [n, K, D]
        flat = W.reshape(-1, W.shape[-1])
        raw = probe.raw(flat).reshape(len(test), K, 2)
        pang = np.degrees(np.arctan2(raw[..., 0], raw[..., 1])) % 360.0
        prad = np.linalg.norm(raw, axis=-1)
        mang = mlp.predict(flat).reshape(len(test), K)
        sign = {"via_plus90": 1.0, "via_minus90": -1.0, "chord": 0.0}[r]
        nominal = (y[:, None] + sign * 180.0 * s[None]) % 360.0
        if r == "chord":
            nominal = np.where(s[None] < 0.5, y[:, None], (y[:, None] + 180.0) % 360.0)
        R = np.zeros((len(test), K))
        for i in range(len(test)):
            vs = ref.snap(nominal[i])
            for j in range(K):
                R[i, j] = mf.nearest_real_agreement(W[i, j], X[i], ref.mean(vs[j], test[i]))
        # midpoint readouts, as offsets from the source
        p_off, m_off = signed_offset(pang[:, mid], y), signed_offset(mang[:, mid], y)
        e = {"probe_angle_offset_from_source_mean_circ": float(np.degrees(np.angle(np.exp(1j * np.radians(p_off)).mean()))),
             "mlp_angle_offset_from_source_mean_circ": float(np.degrees(np.angle(np.exp(1j * np.radians(m_off)).mean()))),
             "probe_err_to_plus90": float(mf.value_error(pang[:, mid], (y + 90) % 360, True).mean()),
             "probe_err_to_minus90": float(mf.value_error(pang[:, mid], (y - 90) % 360, True).mean()),
             "mlp_err_to_plus90": float(mf.value_error(mang[:, mid], (y + 90) % 360, True).mean()),
             "mlp_err_to_minus90": float(mf.value_error(mang[:, mid], (y - 90) % 360, True).mean()),
             "mlp_frac_closer_to_plus90": float((mf.value_error(mang[:, mid], (y + 90) % 360, True)
                                                 < mf.value_error(mang[:, mid], (y - 90) % 360, True)).mean()),
             "probe_frac_closer_to_plus90": float((mf.value_error(pang[:, mid], (y + 90) % 360, True)
                                                   < mf.value_error(pang[:, mid], (y - 90) % 360, True)).mean()),
             "probe_angle_offset_circ_sd_deg": circ_sd_deg(p_off), "mlp_angle_offset_circ_sd_deg": circ_sd_deg(m_off),
             "probe_radius_mid_mean": float(prad[:, mid].mean()), "probe_radius_start_mean": float(prad[:, 0].mean()),
             "probe_radius_min_mean": float(prad.min(1).mean()),
             "nearest_real_R_mid_mean": float(R[:, mid].mean()),
             "end_probe_err_to_target": float(mf.value_error(pang[:, -1], (y + 180) % 360, True).mean()),
             "end_mlp_err_to_target": float(mf.value_error(mang[:, -1], (y + 180) % 360, True).mean()),
             "end_nearest_real_R": float(R[:, -1].mean())}
        if r == "chord":
            Wm = W[:, mid]
            e["nearest_real_R_mid_vs_plus90"] = float(np.mean([mf.nearest_real_agreement(
                Wm[i], X[i], ref.mean(ref.snap(y[i] + 90), test[i])) for i in range(len(test))]))
            e["nearest_real_R_mid_vs_minus90"] = float(np.mean([mf.nearest_real_agreement(
                Wm[i], X[i], ref.mean(ref.snap(y[i] - 90), test[i])) for i in range(len(test))]))
            e["nominal_angle_note"] = "chord R per waypoint: source value for s < 0.5, target after (no nominal angle)"
        out["routes"][r] = {"midpoint": e,
                            "per_waypoint": {"probe_offset_circmean": _circmean_cols(signed_offset(pang, y[:, None])),
                                             "mlp_offset_circmean": _circmean_cols(signed_offset(mang, y[:, None])),
                                             "probe_radius_mean": prad.mean(0).tolist(),
                                             "nearest_real_R_mean": R.mean(0).tolist()}}
        curves[r] = out["routes"][r]["per_waypoint"]
    out["verdict"] = verdict_text(L, out["routes"])
    return out, curves


def _circmean_cols(a):
    return np.degrees(np.angle(np.exp(1j * np.radians(a)).mean(0))).tolist()


def verdict_text(L, R):
    p, m, c = (R[r]["midpoint"] for r in ROUTES)
    ok = (p["mlp_frac_closer_to_plus90"] > 0.8 and m["mlp_frac_closer_to_plus90"] < 0.2)
    collapse = c["probe_radius_mid_mean"] < 0.5 * c["probe_radius_start_mean"]
    return (f"point {L}: at the midpoint the via-+90 spline route reads MLP offset "
            f"{p['mlp_angle_offset_from_source_mean_circ']:+.0f} deg (probe {p['probe_angle_offset_from_source_mean_circ']:+.0f}; "
            f"{p['mlp_frac_closer_to_plus90']:.0%} of carriers closer to theta+90) and the via-+270 route "
            f"{m['mlp_angle_offset_from_source_mean_circ']:+.0f} deg (probe {m['probe_angle_offset_from_source_mean_circ']:+.0f}; "
            f"{1 - m['mlp_frac_closer_to_plus90']:.0%} closer to theta-90): the two routes "
            f"{'pass through opposite sides of the ring, as a ring predicts' if ok else 'do NOT separate cleanly'}. "
            f"Chord midpoint: probe radius {c['probe_radius_mid_mean']:.2f} vs {c['probe_radius_start_mean']:.2f} at the "
            f"source ({'collapsed' if collapse else 'not collapsed'}); midpoint angle circular SD across carriers "
            f"probe {c['probe_angle_offset_circ_sd_deg']:.0f} deg, MLP {c['mlp_angle_offset_circ_sd_deg']:.0f} deg "
            f"(spline routes: MLP {p['mlp_angle_offset_circ_sd_deg']:.0f} / {m['mlp_angle_offset_circ_sd_deg']:.0f} deg). "
            f"Endpoints (theta+180) MLP error: +90 route {p['end_mlp_err_to_target']:.0f}, +270 route "
            f"{m['end_mlp_err_to_target']:.0f}, chord {c['end_mlp_err_to_target']:.0f} deg; nearest-real R at midpoint "
            f"{p['nearest_real_R_mid_mean']:.2f} / {m['nearest_real_R_mid_mean']:.2f} (spline, vs nominal angle), chord "
            f"{c['nearest_real_R_mid_vs_plus90']:.2f} vs theta+90 and {c['nearest_real_R_mid_vs_minus90']:.2f} vs "
            f"theta-90. Exploratory (spline built on all 64 values).")


def plot(all_curves, path, K):
    s = np.linspace(0, 1, K)
    layers = sorted(all_curves)
    fig, axes = plt.subplots(len(layers), 3, figsize=(12, 3.2 * len(layers)), squeeze=False)
    col = {"via_plus90": "#2a6fdb", "via_minus90": "#e07b00", "chord": "#666666"}
    for i, L in enumerate(layers):
        for r in ROUTES:
            c = all_curves[L][r]
            axes[i, 0].plot(s, np.unwrap(np.radians(c["mlp_offset_circmean"])) * 180 / np.pi, color=col[r], label=r)
            axes[i, 0].plot(s, np.unwrap(np.radians(c["probe_offset_circmean"])) * 180 / np.pi, color=col[r], ls=":")
            axes[i, 1].plot(s, c["probe_radius_mean"], color=col[r])
            axes[i, 2].plot(s, c["nearest_real_R_mean"], color=col[r])
        axes[i, 0].set_ylabel(f"point {L}")
        axes[i, 0].set_title("readout offset from source (deg; solid MLP, dotted probe)", fontsize=8)
        axes[i, 1].set_title("linear-probe (sin, cos) radius", fontsize=8)
        axes[i, 2].set_title("nearest-real R vs waypoint's nominal angle", fontsize=8)
        for ax in axes[i]:
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlabel("path fraction s", fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Two-route 180-degree test (exploratory; spline on all 64 values)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(args):
    rdir, fdir = Path(args.results_dir), Path(args.figures_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    out = {"dataset": "direction", "variable": "direction", "label": "exploratory (no held-out target)",
           "k": args.k, "K": args.K, "spline": args.spline,
           "provenance": provenance(args.split, seeds={"mlp": args.seed}, layers=args.layers, k=args.k, K=args.K,
                                    spline=args.spline),
           "design": "sources = all 64 values, target = source + 180 (32 antipodal pairs, both directions); "
                     "carriers = every test clip", "layers": {}}
    curves = {}
    for L in args.layers:
        d = load_inputs("direction", L, "direction", args.act_dir, args.table, args.split)
        res, curves[L] = run_layer(d, L, args)
        res["layer_role"] = layer_role("direction", L)["layer_role"]
        out["layers"][str(L)] = res
        print(res["verdict"], flush=True)
    out["verdict"] = " ".join(out["layers"][str(L)]["verdict"] for L in args.layers)
    tag = "_".join(f"L{L}" for L in args.layers)
    (rdir / f"p2_two_route_direction_{tag}.json").write_text(json.dumps(out, indent=1))
    plot(curves, fdir / f"fig4e_two_route_direction_{tag}.png", args.K)
    return out


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, nargs="+", default=[12, 22])
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--spline", default="smooth", choices=("interp", "smooth"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--act-dir", default=None)
    p.add_argument("--table", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--figures-dir", default=str(PROJECT_ROOT / "figures"))
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse())

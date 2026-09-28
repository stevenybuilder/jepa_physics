"""Goodfire A.5 isometry test with its linear baseline, and with behaviour spaces that are not built from the steered
activations (PART2_SECOND_LOOK.md section B). CPU only, stored activations.

Goodfire §2.3 / A.5: Pearson r between pairwise geodesic distances on the activation manifold M_h and on the behaviour
manifold M_y (r = 0.89-0.999 across their tasks), against r between straight-line (chord) distances in activation
space and the same M_y geodesics (r = 0.36-0.89; 0.06 in the mountain-car world model). run_part2 stores the geodesic
r (isometry_probe, isometry_behaviour) but never the chord baseline, and its M_y (Eq. 9 at the steered layer) is a
function of the steered activations.

Here, per direction layer, on the 64 direction values (W = 64 vertices, K = 0 interior points as Goodfire uses for
W >= 81; A.5's shared-geodesic exclusion is then empty):
  activation side (knot folds 0-2, PCA-64 fit on them): the count-weighted periodic smoothing spline that run_part2
      steers with, and Goodfire's interpolating periodic spline; intrinsic angle chosen as in run_part2
      (choose_angle_source). d_geo = arc length along the spline (short way round); d_lin = Euclidean distance
      between the same vertex points in PCA-64.
  behaviour side (probe folds 3-4, disjoint clips), each a periodic smoothing spline through per-value centroids
      with the true angle as coordinate, distances = arc length:
      eq9_same_layer   Goodfire Eq. 9 M_y at the steered layer (Hellinger geodesics; circular, for reference)
      encoder_out      the encoder's last point (25, post-LN) activations, PCA-64
      predictor        the V-JEPA 2 predictor's own unedited forecast of tubelets 4-7 from frames 1-8
                       (artifacts/session2/native/pred_pooled_all.npy, mean over the 4 steps), PCA-64
      concept          the true angular distance |d theta| (the conceptual metric d_Z)
  CIs: 200 bootstrap draws resampling clips within each value on both sides (PCA bases fixed).

  python scripts/run_isometry_linear.py --layers 8 12 22
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from wm import manifold as mf
from wm.data import PROJECT_ROOT
from wm.p2_data import load_inputs
from wm.provenance import layer_role, provenance

PRED = PROJECT_ROOT / "artifacts" / "session2" / "native" / "pred_pooled_all.npy"


def geo_matrix(curve, n=4000):
    """Pairwise arc length between the curve's vertices (curve.coords), short way round, from one dense table."""
    tt, s = mf.arc_length_table(curve, n)
    total = s[-1]
    lo = curve.t_range()[0]
    t = (curve.coords - lo) % mf.TWO_PI + lo if curve.periodic else curve.coords
    sv = np.interp(t, tt, s)
    D = np.abs(sv[:, None] - sv[None])
    return np.minimum(D, total - D) if curve.periodic else D


def by_value(G, curve_values, values):
    """Reorder a vertex-ordered matrix into sorted-value order."""
    idx = {float(v): i for i, v in enumerate(curve_values)}
    o = [idx[float(v)] for v in values]
    return G[np.ix_(o, o)]


def act_curves(Z, y, rng=None):
    """Smoothing (run_part2) and interpolating (Goodfire A.3) periodic splines through per-value centroids of Z."""
    if rng is not None:
        Z, y = resample(Z, y, rng)
    cent = mf.centroids(Z, y)
    choice = mf.choose_angle_source(cent["C"], cent["values"])
    out = {}
    for sp in ("smooth", "interp"):
        c = mf.fit_curve(cent, True, angle=choice["angle"], plane=choice["plane"], spline=sp)
        G = by_value(geo_matrix(c), c.values, cent["values"])
        P = by_value(np.linalg.norm(c.points[:, None] - c.points[None], axis=-1), c.values, cent["values"])
        out[sp] = {"geo": G, "lin": P}
    return out, cent["values"], choice


def beh_curve(Z, y, rng=None):
    if rng is not None:
        Z, y = resample(Z, y, rng)
    cent = mf.centroids(Z, y)
    c = mf.fit_curve(cent, True, angle="labels", spline="smooth")
    return by_value(geo_matrix(c), c.values, cent["values"])


def resample(Z, y, rng):
    idx = np.concatenate([rng.choice(np.flatnonzero(y == v), (y == v).sum()) for v in np.unique(y)])
    return Z[idx], y[idx]


def corr(A, B):
    iu = np.triu_indices(len(A), 1)
    return float(np.corrcoef(A[iu], B[iu])[0, 1]), float(spearmanr(A[iu], B[iu])[0])


def run_layer(L, n_boot, seed=0):
    d = load_inputs("direction", L)
    knot, probe = d["role"] == "knot", d["role"] == "probe"
    y = d["y"]
    pca = mf.fit_pca(d["X"][knot], 64)
    Zk = pca.project(d["X"][knot])
    d25 = load_inputs("direction", 25)
    pca25 = mf.fit_pca(d25["X"][probe], 64)
    Z25 = pca25.project(d25["X"][probe])
    pred = np.load(PRED).astype(np.float64).mean(1)
    pcap = mf.fit_pca(pred[probe], 64)
    Zp = pcap.project(pred[probe])
    yk, yp = y[knot], y[probe]
    values = np.unique(y)

    # Eq. 9 M_y at the steered layer: probe-fold reference smoothing spline in the knot PCA (as run_part2 builds it)
    ref_curve = mf.fit_curve(mf.centroids(pca.project(d["X"][probe]), yp), True, angle="labels", spline="smooth")
    bm = mf.BehaviourManifold(d["X"][probe], yp, True, curve=ref_curve, pca=pca)
    eq9 = bm.geodesic_matrix(values)
    dth = np.abs(((values[:, None] - values[None]) + 180) % 360 - 180)

    def all_corrs(act, beh):
        out = {}
        for sp, M in act.items():
            for kind in ("geo", "lin"):
                for name, B in beh.items():
                    r, rho = corr(M[kind], B)
                    out[f"{sp}.{kind}.{name}"] = {"pearson": r, "spearman": rho}
        return out

    act, vals, choice = act_curves(Zk, yk)
    assert np.allclose(vals, values)
    beh = {"eq9_same_layer": eq9, "encoder_out": beh_curve(Z25, yp), "predictor": beh_curve(Zp, yp), "concept": dth}
    point = all_corrs(act, beh)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        a, _, _ = act_curves(Zk, yk, rng)
        b = {"eq9_same_layer": eq9, "encoder_out": beh_curve(Z25, yp, rng), "predictor": beh_curve(Zp, yp, rng),
             "concept": dth}
        boots.append(all_corrs(a, b))
    res = {}
    for key, v in point.items():
        pb = np.array([b[key]["pearson"] for b in boots])
        res[key] = {**v, "pearson_ci95": [float(np.percentile(pb, 2.5)), float(np.percentile(pb, 97.5))]}
    diffs = {}
    for sp in act:
        for name in beh:
            dd = np.array([b[f"{sp}.geo.{name}"]["pearson"] - b[f"{sp}.lin.{name}"]["pearson"] for b in boots])
            diffs[f"{sp}.{name}"] = {"geo_minus_lin": point[f"{sp}.geo.{name}"]["pearson"]
                                     - point[f"{sp}.lin.{name}"]["pearson"],
                                     "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}
    return {**layer_role("direction", L, "direction"), "angle_choice": {k: choice[k] for k in ("angle", "plane")},
            "n_knot_clips": int(knot.sum()), "n_probe_clips": int(probe.sum()), "correlations": res,
            "geo_minus_lin": diffs}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, nargs="+", default=[8, 12, 22])
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--out", default=str(PROJECT_ROOT / "results" / "p2_isometry_linear.json"))
    args = p.parse_args(argv)
    out = {"provenance": provenance(None, seeds={"bootstrap": 0}, layers=args.layers, pool="meanpool", k=64),
           "method": ("Goodfire A.5: Pearson r over the upper triangle of 64 x 64 pairwise distance matrices; "
                      "key = activation_spline.distance.behaviour_space; activation_spline smooth = run_part2's "
                      "count-weighted smoothing spline, interp = Goodfire's interpolating spline; distance geo = arc "
                      "length along it, lin = chord in PCA-64 between the same vertices"),
           "caveat": ("eq9_same_layer is built from the steered layer's own activations (circular, as in Goodfire's "
                      "mountain-car B.1); encoder_out, predictor and concept are not. For a planar circle the chord "
                      "is a monotone function of the arc (2R sin(d/2)), so Pearson geo-vs-lin gaps on a ring are "
                      "small by construction; Spearman is reported beside Pearson."),
           "layers": {}}
    for L in args.layers:
        out["layers"][str(L)] = run_layer(L, args.n_boot)
        r = out["layers"][str(L)]["correlations"]
        print(f"L{L}: " + "  ".join(f"{b}: geo {r[f'smooth.geo.{b}']['pearson']:.3f} lin {r[f'smooth.lin.{b}']['pearson']:.3f}"
                                     for b in ("eq9_same_layer", "encoder_out", "predictor", "concept")), flush=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

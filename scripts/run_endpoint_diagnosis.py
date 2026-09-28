"""Why does the smoothing spline lose the held-out endpoint to the raw-centroid chord on the direction ring?

Diagnosis for REPORT section 4.3 (the "chord beats spline at held-out endpoints" negative). For each held-out arc
(contiguous design, seed = arc) it rebuilds exactly what scripts/run_part2.py builds (p2.build: PCA-64 on knot folds
at kept values, centroids, the count-weighted smoothing spline, the raw-centroid polyline) and then steers the same
picked test clips (p2.pick_clips) toward each held-out value with a family of AIM POINTS P(v) in PCA-64:

  spline_smooth    our curve (--spline smooth), the stored "manifold" arm
  chord_raw        chord between the raw kept centroids at the value fraction, the stored "linear_raw" arm (A.9)
  chord_smooth     chord between the smoothed knots, the stored "linear" arm
  spline_interp    interpolating periodic cubic through the raw kept centroids (Goodfire A.3; scipy)
  causalab_interp  the authors' own code: causalab SplineManifold (CubicSpline1D, periodic, smoothness=0) on the
                   same knots and coordinates, decoded at the same held-out coordinate
  causalab_lam_*   causalab Reinsch smoothing, lambda sweep
  splrep_s*        our SmoothSpline with its smoothing s = f * m (f = 1 is ours), f swept
  oracle           the TRUE held-out centroid (knot-fold clips at the held-out value, never used by any arm);
                   raw centroid at kept values
  hybrids          target from one rule, source anchor from another (decomposes the additive edit)

Edit rules: "add" (ours: x + lift(P(tgt) - P(src)), residual kept) and "replace" (Goodfire A.6: PCA-64 part of x
replaced by P(tgt), residual kept); norm variants rescale one arm's additive edit to another arm's per-clip norm.
Readers: the held-out ridge probe (probe folds), the MLP evaluator (probe folds, seed 0), nearest-real R (test clips).

  python scripts/run_endpoint_diagnosis.py --layer 22 --seeds 0 1 2 ... --out results/_diag_L22.json
"""
import argparse
import importlib
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

from wm import bakeoff as bo
from wm import manifold as mf
from wm.data import PROJECT_ROOT
from wm.p2_data import load_inputs

_spec = importlib.util.spec_from_file_location("run_part2", Path(__file__).with_name("run_part2.py"))
p2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p2)

# the authors' spline code, loaded file-by-file (causalab/methods/__init__ pulls in pyvene; the spline files need
# only torch)
_CL = PROJECT_ROOT / "refs" / "causalab" / "causalab" / "methods" / "spline"
_pkg = types.ModuleType("cl_spline")
_pkg.__path__ = [str(_CL)]
sys.modules["cl_spline"] = _pkg
cl_manifold = importlib.import_module("cl_spline.manifold")
TWO_PI = 2 * np.pi

LAMBDAS = (1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
CV_LAMBDAS = (0.0,) + LAMBDAS + (1e-3,)


def cv_lambda(coords, Ck, values, block=8, margin=4):
    """Choose causalab's Reinsch smoothness by leave-block-out on the KEPT knots only (no test clip, no held-out
    value): knots in value order after the real gap; blocks of `block` consecutive kept values at least `margin`
    knots from the real gap are removed in turn, the spline refit on the rest, and its points at the removed knots'
    coordinates compared with their raw centroids. The chord between the block's neighbours is scored the same way.
    Returns dict(lambda_cv, cv_err per lambda, cv_err_chord)."""
    v = np.asarray(values, float)
    # kept values in circular value order, starting just after the real gap
    order = np.argsort(v)
    gaps = np.diff(np.append(v[order], v[order][0] + 360.0))
    start = (int(np.argmax(gaps)) + 1) % len(v)
    seq = np.roll(order, -start)
    errs = {lam: [] for lam in CV_LAMBDAS}
    chord = []
    j = margin
    while j + block <= len(seq) - margin:
        out = seq[j:j + block]
        keep = np.setdiff1d(np.arange(len(v)), out)
        a, b = seq[j - 1], seq[j + block]
        for lam in CV_LAMBDAS:
            f, _ = causalab_curve(coords[keep], Ck[keep], lam)
            errs[lam].append(np.linalg.norm(f(coords[out]) - Ck[out], axis=1).mean())
        fr = ((v[out] - v[a]) % 360.0) / ((v[b] - v[a]) % 360.0)
        chord.append(np.linalg.norm((1 - fr)[:, None] * Ck[a] + fr[:, None] * Ck[b] - Ck[out], axis=1).mean())
        j += block
    cv = {f"{lam:g}": float(np.mean(e)) for lam, e in errs.items()}
    best = min(CV_LAMBDAS, key=lambda lam: np.mean(errs[lam]))
    return {"lambda_cv": float(best), "cv_err_by_lambda": cv, "cv_err_chord": float(np.mean(chord)),
            "n_blocks": len(chord)}
S_FACTORS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0)


def causalab_curve(coords, C, smoothness=0.0, periods=(TWO_PI,)):
    """causalab SplineManifold on knot coordinates (radians) and PCA-64 knots; returns decode(u) as numpy."""
    cp = torch.tensor(np.asarray(coords, float)[:, None], dtype=torch.float64)
    tp = torch.tensor(np.asarray(C, float), dtype=torch.float64)
    man = cl_manifold.SplineManifold(cp, tp, intrinsic_dim=1, ambient_dim=tp.shape[1], smoothness=smoothness,
                                     periodic_dims=[0], periods=None if periods is None else list(periods),
                                     spline_method="cubic")

    def f(u):
        u = np.asarray(u, float)
        out = man.decode(torch.tensor(u.reshape(-1, 1), dtype=torch.float64)).numpy()
        return out.reshape(u.shape + (tp.shape[1],))
    return f, man


def foreign_knots(curve, v):
    """Knots of OTHER values lying on the spline stretch between v's value neighbours a, b (short way in coordinate
    order), for a held-out value v; issue #265."""
    a, b, f = curve.value_neighbours(np.atleast_1d(v))
    a, b = int(a[0]), int(b[0])
    ta, tb = curve.coords[a], curve.coords[b]
    step = float(mf.wrap_pi(tb - ta))
    rel = mf.wrap_pi(curve.coords - ta)
    inside = (rel * np.sign(step) > 1e-12) & (np.abs(rel) < abs(step) - 1e-12)
    inside[[a, b]] = False
    return int(inside.sum())


def run_arc(d, L, seed, angle, readers, aim="coord", n_clips=48):
    t0 = time.time()
    m = p2.build(d, 64, angle, "contiguous", seed, 0, "smooth", aim=aim)
    pca, cent, curve, raw = m["pca"], m["cent"], m["curve"], m["raw_curve"]
    held = m["held"]
    picks = p2.pick_clips(d, held, n_clips, seed)
    # true held-out centroids: knot-fold clips at the held-out values (excluded from every fit)
    knot_all = d["role"] == "knot"
    true_C = {float(v): pca.project(d["X"][knot_all & np.isclose(d["y"], v)].astype(float)).mean(0) for v in held}
    raw_C = {float(v): c for v, c in zip(cent["values"], cent["C"])}
    n_true = {float(v): int((knot_all & np.isclose(d["y"], v)).sum()) for v in held}

    # knots in the curve's order, raw centroids
    Ck = np.array([raw_C[float(v)] for v in curve.values])
    cof = curve.coord_of_value

    # interpolating spline (ours, scipy) and the authors' code on the same knots/coordinates
    ci = mf.periodic_cubic(Ck, curve.values, angle=curve.coords, smooth=None)
    ci.aim = aim
    cl_interp, cl_man = causalab_curve(curve.coords, Ck, 0.0)
    cl_inferred, cl_man_inf = causalab_curve(curve.coords, Ck, 0.0, periods=None)   # causalab's default period rule

    P = {}
    P["spline_smooth"] = lambda v: curve(cof(v))
    P["chord_raw"] = lambda v: mf.piecewise_linear_point(raw, v)
    P["chord_smooth"] = lambda v: mf.piecewise_linear_point(curve, v)
    P["spline_interp"] = lambda v: ci(ci.coord_of_value(v))
    P["causalab_interp"] = lambda v: cl_interp(cof(v))
    P["causalab_interp_inferred_period"] = lambda v: cl_inferred(cof(v))
    for lam in LAMBDAS:
        f, _ = causalab_curve(curve.coords, Ck, lam)
        P[f"causalab_lam_{lam:g}"] = (lambda f: (lambda v: f(cof(v))))(f)
    for sf in S_FACTORS:
        sp = SmoothSplineS(curve.coords, Ck, np.array([cent["count"][list(cent["values"]).index(v)]
                                                       for v in curve.values]), cent["sd_coord"], sf)
        P[f"splrep_s{sf:g}"] = (lambda sp: (lambda v: sp(cof(v))))(sp)

    cvl = cv_lambda(np.asarray(curve.coords, float), Ck, curve.values)
    f_cv, _ = causalab_curve(curve.coords, Ck, cvl["lambda_cv"])
    P["causalab_lam_cv"] = lambda v: f_cv(cof(v))
    # chord through the Reinsch-smoothed knots: denoised anchors without the curve's bend between them
    cv_knots = mf.Curve(spline=None, values=curve.values, coords=curve.coords, points=f_cv(curve.coords),
                        periodic=True, coord_source=curve.coord_source, kind="raw_knots", aim=curve.aim)
    P["chord_lam_cv_knots"] = lambda v: mf.piecewise_linear_point(cv_knots, v)

    def oracle(v):
        v = np.atleast_1d(np.asarray(v, float))
        return np.array([true_C.get(float(x), raw_C.get(float(x))) for x in v])
    P["oracle"] = oracle
    # label-free coordinate as causalab computes it (PCA mode), interpolating, held-out coordinate by value order
    gp = mf.goodfire_periodic_angle(Ck)
    try:
        cg = mf.periodic_cubic(Ck, curve.values, angle=gp["angle"], smooth=None)
        cg.aim = "coord"
        P["causalab_pcaangle_interp"] = lambda v: cg(cg.coord_of_value(v))
    except ValueError:
        pass

    # ---------------- aim geometry at the held-out targets ----------------
    hv = np.asarray(held, float)
    mu = np.array([true_C[float(v)] for v in hv])
    aim_geo = {}
    for name, fn in P.items():
        pts = fn(hv)
        pred = readers["probe"].predict(pca.lift(pts))
        aim_geo[name] = {"dist_to_true_centroid": float(np.linalg.norm(pts - mu, axis=1).mean()),
                         "probe_err_of_aim_point": float(mf.value_error(pred, hv, True).mean())}
    # probe reading of the true centroid itself (the reader's own floor at an aim point)
    Sp, Cp = P["spline_interp"](hv), P["chord_raw"](hv)
    Ss = P["spline_smooth"](hv)
    beta = [float(np.dot(u - c, s - c) / max(np.dot(s - c, s - c), 1e-12)) for u, c, s in zip(mu, Cp, Sp)]
    beta_s = [float(np.dot(u - c, s - c) / max(np.dot(s - c, s - c), 1e-12)) for u, c, s in zip(mu, Cp, Ss)]
    # knot displacement of the smoothing spline (smoothed knot vs raw centroid)
    knot_disp = float(np.linalg.norm(curve.points - Ck, axis=1).mean())
    # kept knots: probe reading of the raw centroid, the smoothed knot and an independent (probe-fold) centroid, and
    # whether smoothing moves knots toward the independent centroid (denoising) or away
    pr_rows = d["role"] == "probe"
    kv = np.asarray(curve.values, float)
    ind = np.array([pca.project(d["X"][pr_rows & np.isclose(d["y"], v)].astype(float)).mean(0) for v in kv])
    kept_geo = {}
    for name, pts in (("raw_centroid", Ck), ("smoothed_knot", curve.points), ("probe_fold_centroid", ind),
                      ("causalab_lam_cv_knot", f_cv(curve.coords))):
        kept_geo[name] = {"probe_err": float(mf.value_error(readers["probe"].predict(pca.lift(pts)), kv, True).mean()),
                          "dist_to_probe_fold_centroid": float(np.linalg.norm(pts - ind, axis=1).mean())}
    # the authors' code vs ours on the same knots: dense curve comparison
    tt = np.linspace(0, TWO_PI, 721)
    diff_cl = float(np.abs(cl_interp(tt) - ci(tt)).max())
    arc_geo = {"n_true_centroid_clips": n_true, "bulge_beta_interp": beta, "bulge_beta_smooth": beta_s,
               "bulge_note": "beta = <mu - chord, spline - chord> / |spline - chord|^2 at each held-out target: "
                             "1 = the true held-out centroid sits where the spline puts it, 0 = on the raw chord",
               "sagitta_interp_vs_rawchord": float(np.linalg.norm(Sp - Cp, axis=1).mean()),
               "sagitta_smooth_vs_rawchord": float(np.linalg.norm(Ss - Cp, axis=1).mean()),
               "sagitta_lam_cv_vs_rawchord": float(np.linalg.norm(P["causalab_lam_cv"](hv) - Cp, axis=1).mean()),
               "sagitta_lam_cv_vs_its_own_chord": float(np.linalg.norm(P["causalab_lam_cv"](hv)
                                                                       - P["chord_lam_cv_knots"](hv), axis=1).mean()),
               "bulge_beta_lam_cv": [float(np.dot(u - c, s - c) / max(np.dot(s - c, s - c), 1e-12))
                                     for u, c, s in zip(mu, Cp, P["causalab_lam_cv"](hv))],
               "centroid_noise_true_heldout": float(np.mean([np.sqrt((pca.project(
                   d["X"][knot_all & np.isclose(d["y"], v)].astype(float)).var(0, ddof=1)).sum() / n_true[float(v)])
                   for v in hv])),
               "smoothing_knot_displacement_mean": knot_disp,
               "causalab_vs_scipy_interp_max_abs_diff": diff_cl,
               "causalab_inferred_period": float(cl_man_inf.periods[0]),
               "goodfire_pca_angle_passes": gp["passes"],
               "foreign_knots_per_target": [foreign_knots(curve, v) for v in hv],
               "angle_source": curve.coord_source, "aim": aim, "kept_knots": kept_geo, "lambda_cv": cvl}

    # ---------------- steering: every aim rule, edit rule and reader ----------------
    acc = {}

    def add(name, xs, x, tgt):
        a = acc.setdefault(name, {"probe": [], "mlp": [], "R": [], "norm": []})
        a["probe"].append(mf.value_error(readers["probe"].predict(xs), tgt, True))
        a["mlp"].append(mf.value_error(readers["mlp"].predict(xs), tgt, True))
        a["R"].append(readers["near"].score(xs, x, tgt))
        a["norm"].append(np.linalg.norm(xs - x, axis=1))

    cl_edit = {}
    for tgt, pick in picks.items():
        x, src = d["X"][pick].astype(float), d["y"][pick]
        Z, resid = pca.project(x), pca.complement(x)
        tv = np.full(len(src), tgt)
        cache = {n: (fn(tv), fn(src)) for n, fn in P.items()}
        dZ = {n: pt - ps for n, (pt, ps) in cache.items()}
        for n in P:
            add(f"add:{n}", x + pca.lift_delta(dZ[n]), x, tgt)
            add(f"replace:{n}", pca.lift(np.broadcast_to(cache[n][0], Z.shape)) + resid, x, tgt)
        # hybrids: target aim of one rule, source anchor of another
        for tname, sname in (("spline_smooth", "chord_raw"), ("chord_raw", "spline_smooth"),
                             ("spline_interp", "chord_raw"), ("oracle", "spline_smooth"),
                             ("spline_smooth", "oracle"), ("causalab_lam_cv", "chord_raw"),
                             ("chord_raw", "causalab_lam_cv")):
            add(f"hybrid:tgt={tname},src={sname}", x + pca.lift_delta(cache[tname][0] - cache[sname][1]), x, tgt)
        # norm variants
        nz = lambda v: np.linalg.norm(v, axis=1, keepdims=True)
        for a_name, ref in (("chord_raw", "spline_smooth"), ("spline_smooth", "chord_raw"),
                            ("spline_interp", "chord_raw"), ("chord_raw", "spline_interp"),
                            ("oracle", "spline_smooth"), ("oracle", "chord_raw")):
            dz = dZ[a_name] * np.where(nz(dZ[a_name]) > 0, nz(dZ[ref]) / np.maximum(nz(dZ[a_name]), 1e-12), 0.0)
            add(f"add_norm:{a_name}@{ref}", x + pca.lift_delta(dz), x, tgt)
        # Goodfire's full-space linear baseline (whole activation replaced by the full-space chord point)
        full_pt = mf.piecewise_linear_point(m["full_curve"], tv)
        add("replace_full:chord_raw_fullspace", full_pt, x, tgt)
        # authors' code vs ours: the edit vector for the causalab interp arm vs our scipy interp arm, and vs smooth
        for other in ("spline_interp", "spline_smooth", "chord_raw"):
            e1, e2 = dZ["causalab_interp"], dZ[other]
            cos = (e1 * e2).sum(1) / np.maximum(np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1), 1e-12)
            cl_edit.setdefault(other, {"cos": [], "norm_ratio": [], "endpoint_dist": []})
            cl_edit[other]["cos"].append(cos)
            cl_edit[other]["norm_ratio"].append(np.linalg.norm(e1, axis=1) / np.maximum(np.linalg.norm(e2, axis=1),
                                                                                        1e-12))
            cl_edit[other]["endpoint_dist"].append(np.linalg.norm(cache["causalab_interp"][0] - cache[other][0],
                                                                  axis=1))
    arms = {}
    for n, a in acc.items():
        arms[n] = {"probe_err": float(np.concatenate(a["probe"]).mean()),
                   "mlp_err": float(np.concatenate(a["mlp"]).mean()),
                   "nearest_real_R": float(np.concatenate(a["R"]).mean()),
                   "delta_norm": float(np.concatenate(a["norm"]).mean())}
    # paired per-clip gaps vs the raw chord (additive), all readers
    ref = acc["add:chord_raw"]
    gaps = {}
    for n, a in acc.items():
        g = {}
        for r in ("probe", "mlp", "R"):
            diff = np.concatenate(a[r]) - np.concatenate(ref[r])
            g[r] = float(diff.mean())
        gaps[n] = g
    cl_cmp = {o: {k: float(np.concatenate(v).mean()) for k, v in dd.items()} for o, dd in cl_edit.items()}
    return {"layer": L, "seed": seed, "angle": angle, "aim": aim, "held_out_values": hv.tolist(),
            "n_steered": int(sum(len(p) for p in picks.values())), "arc_geometry": arc_geo, "aim_geometry": aim_geo,
            "arms": arms, "gap_minus_add_chord_raw": gaps, "causalab_edit_vs": cl_cmp,
            "seconds": round(time.time() - t0, 1)}


class SmoothSplineS(mf.SmoothSpline):
    """mf.SmoothSpline with its smoothing s scaled by a factor (s = factor * m; factor 1 = ours; 0 = interpolation
    through every knot with the same splrep machinery)."""

    def __init__(self, coord, P, count, sd, factor):
        from scipy.interpolate import splrep
        coord, P = np.asarray(coord, float), np.asarray(P, float)
        w = np.sqrt(np.asarray(count, float))
        self.periodic, self.t0 = True, float(coord[0])
        coord, P, w = np.append(coord, coord[0] + TWO_PI), np.vstack([P, P[:1]]), np.append(w, w[0])
        m = len(coord) - 1
        self.tck = [splrep(coord, P[:, c], w=w / max(sd[c], 1e-12), k=3, s=float(factor * m), per=1)
                    for c in range(P.shape[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--angle", default="auto", choices=("auto", "labels", "unsupervised"))
    ap.add_argument("--aim", default="coord", choices=("coord", "arc"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = load_inputs("direction", a.layer, "direction")
    pr = d["role"] == "probe"
    test = np.flatnonzero(d["role"] == "test")
    readers = {"probe": mf.ProbeReadout(d["X"][pr], d["y"][pr], True),
               "mlp": bo.MLPReadout(d["X"][pr], d["y"][pr], True, 0),
               "near": mf.NearestRealReadout(d["X"][test], d["y"][test])}
    angle = a.angle if a.angle != "auto" else ("labels" if a.layer == 22 else "unsupervised")
    out = []
    for s in a.seeds:
        r = run_arc(d, a.layer, s, angle, readers, aim=a.aim)
        out.append(r)
        print(a.layer, s, angle, a.aim, r["seconds"], "s  spline", round(r["arms"]["add:spline_smooth"]["probe_err"], 2),
              "chord", round(r["arms"]["add:chord_raw"]["probe_err"], 2), flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

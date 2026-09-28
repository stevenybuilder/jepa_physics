"""Path stage of the endpoint diagnosis (ledger #307): the PATH metrics of the same splines whose endpoints
scripts/run_endpoint_diagnosis.py scores.

Per held-out contiguous arc (seed = arc, p2.build exactly as run_endpoint_diagnosis.run_arc builds it: PCA-64 on knot
folds, value-ordered knots, raw kept centroids, the stored count-weighted FITPACK smoother) and per picked test clip
(p2.pick_clips, 48 per held-out target), four aim curves are walked from source to target:

  chord_raw            the raw-centroid chord (run_part2.subspace_arms "linear_raw": mf.linear_coords)
  fitpack_smooth       the stored FITPACK smoothing spline (run_part2.subspace_arms "manifold": mf.manifold_coords,
                       mode="shift")
  interp_causalab_lam0 causalab SplineManifold, periodic cubic, smoothness 0 (interpolating), on the same knots and
                       coordinates (run_endpoint_diagnosis.causalab_curve), walked with mf.manifold_coords
  causalab_lam_cv      causalab Reinsch smoother, lambda by leave-block-out on kept knots (run_endpoint_diagnosis.cv_lambda)

The walk is mf.manifold_coords' rule for every curve: t_k = ta + s_k * curve.step(ta, tb), s = linspace(0, 1, K),
z_k = z + f(t_k) - f(ta) (additive, residual kept, lifted with run_part2.compose). The spline arms all use the
stored curve's coordinates and step (the short way round in the intrinsic angle), so they traverse the same arc.

Per waypoint: probe (sin, cos) radius |ProbeReadout.raw| (run_part2.evaluate lines "wp_radius"), the probe's decoded
angle, nearest-real R against the PROBE-FOLD full-space centroid at the waypoint's intended angle
(src + s_k * traversed shift, snapped to the nearest probe-fold value), and the signed progress of the decoded angle
along the traversed arc. Also the endpoint R against test clips at the target (run_part2.evaluate's nearest_real_R).
Norm variants: each spline arm at its own norm and rescaled per clip and waypoint to the chord's ||delta|| (the
run_part2 linear_dose_matched rule, reversed).

  python scripts/run_endpoint_diagnosis_path.py --layer 12 --seeds 0 ... 15 --K 11 --out results/endpoint_diagnosis_raw/path_L12_K11.json
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from wm import manifold as mf
from wm.p2_data import load_inputs

_here = Path(__file__).parent
_s = importlib.util.spec_from_file_location("run_endpoint_diagnosis", _here / "run_endpoint_diagnosis.py")
ed = importlib.util.module_from_spec(_s)
_s.loader.exec_module(ed)
p2 = ed.p2

ARMS = ("chord_raw", "fitpack_smooth", "interp_causalab_lam0", "causalab_lam_cv")


class FnCurve:
    """Aim function f(t) on the stored curve's coordinates, with the stored curve's step, for mf.manifold_coords."""

    def __init__(self, f, base):
        self.f, self.step, self.periodic = f, base.step, base.periodic

    def __call__(self, t):
        return self.f(t)


def path_metrics(W, x, src, tgt, s, shift_signed, probe, near, pf_vals, pf_C):
    """W [n, K, D] full-space waypoints. Returns per-clip, per-waypoint arrays."""
    n, K = W.shape[:2]
    raw = probe.raw(W.reshape(n * K, -1)).reshape(n, K, 2)
    radius = np.linalg.norm(raw, axis=-1)
    ang = np.degrees(np.arctan2(raw[..., 0], raw[..., 1])) % 360.0
    sign = np.sign(shift_signed)[:, None]
    # signed progress along the traversed arc: decoded angle relative to source, wrapped to (-180, 180], then unwrapped
    rel = (ang - src[:, None] + 180.0) % 360.0 - 180.0
    prog = np.unwrap(np.radians(rel), axis=1)
    prog = np.degrees(prog) * sign
    steps = np.diff(prog, axis=1)
    mono = (steps > 0).mean(1)
    from scipy.stats import spearmanr
    rho = np.array([spearmanr(np.arange(K), p).correlation if np.ptp(p) > 0 else 0.0 for p in prog])
    # intended angle at each waypoint, snapped to the nearest probe-fold value
    want = (src[:, None] + s[None, :] * shift_signed[:, None]) % 360.0
    dd = np.abs((want[..., None] - pf_vals[None, None, :] + 180.0) % 360.0 - 180.0)
    ref = pf_C[dd.argmin(-1)]                                              # [n, K, D]
    R_wp = mf.nearest_real_agreement(W, x[:, None], ref)
    R_end = near.score(W[:, -1], x, tgt)
    err_end = mf.value_error(ang[:, -1], tgt, True)
    return {"radius": radius, "angle": ang, "progress": prog, "monotone_frac": mono, "ordering_spearman": rho,
            "R_wp": R_wp, "R_end_test": R_end, "err_end": err_end, "radius_min": radius.min(1),
            "radius_mean": radius.mean(1), "R_wp_mean": R_wp.mean(1)}


def run_arc(d, L, seed, angle, probe, near, pf_vals, pf_C, Ks, n_clips=48):
    t0 = time.time()
    m = p2.build(d, 64, angle, "contiguous", seed, 0, "smooth", aim="coord")
    pca, cent, curve, raw_c = m["pca"], m["cent"], m["curve"], m["raw_curve"]
    held = m["held"]
    picks = p2.pick_clips(d, held, n_clips, seed)
    raw_C = {float(v): c for v, c in zip(cent["values"], cent["C"])}
    Ck = np.array([raw_C[float(v)] for v in curve.values])
    f_int, _ = ed.causalab_curve(curve.coords, Ck, 0.0)
    cvl = ed.cv_lambda(np.asarray(curve.coords, float), Ck, curve.values)
    f_cv, _ = ed.causalab_curve(curve.coords, Ck, cvl["lambda_cv"])
    curves = {"fitpack_smooth": curve, "interp_causalab_lam0": FnCurve(f_int, curve),
              "causalab_lam_cv": FnCurve(f_cv, curve)}
    out = {K: {} for K in Ks}
    meta = {"src": [], "tgt": [], "clip": []}
    for tgt, pick in picks.items():
        x, src = d["X"][pick].astype(float), d["y"][pick].astype(float)
        Z, resid = pca.project(x), pca.complement(x)
        arc = p2.traversed_arc(curve, src, tgt)                           # +1 / -1 label orientation of the walk
        shift_signed = arc * (((tgt - src) * arc) % 360.0)
        meta["src"].append(src); meta["tgt"].append(np.full(len(src), tgt)); meta["clip"].append(pick)
        ta, tb = curve.coord_of_value(src), curve.coord_of_value(np.full(len(src), tgt))
        for K in Ks:
            s = np.linspace(0.0, 1.0, K)
            Zp = {"chord_raw": mf.linear_coords(Z, mf.piecewise_linear_point(raw_c, src),
                                                mf.piecewise_linear_point(raw_c, np.full(len(src), tgt)), K)}
            for a, c in curves.items():
                Zp[a] = mf.manifold_coords(Z, c, ta, tb, K, "shift")
            dl = Zp["chord_raw"] - Z[:, None]
            nl = np.linalg.norm(dl, axis=-1, keepdims=True)
            for a in list(curves):
                da = Zp[a] - Z[:, None]
                na = np.linalg.norm(da, axis=-1, keepdims=True)
                Zp[f"{a}@chord_norm"] = Z[:, None] + da * np.where(na > 0, nl / np.where(na > 0, na, 1.0), 0.0)
            for a, zp in Zp.items():
                W = p2.compose(pca, zp, resid)
                r = path_metrics(W, x, src, tgt, s, shift_signed, probe, near, pf_vals, pf_C)
                r["delta_norm_pca"] = np.linalg.norm(zp - Z[:, None], axis=-1)
                acc = out[K].setdefault(a, {})
                for q, v in r.items():
                    acc.setdefault(q, []).append(v)
    res = {"layer": L, "seed": seed, "angle": angle, "held_out_values": [float(v) for v in held],
           "lambda_cv": cvl["lambda_cv"], "clip": np.concatenate(meta["clip"]).tolist(),
           "src": np.concatenate(meta["src"]).tolist(), "tgt": np.concatenate(meta["tgt"]).tolist(), "K": {}}
    for K in Ks:
        res["K"][str(K)] = {a: {q: np.concatenate(v).tolist() for q, v in dd.items() if q not in ("angle",)}
                            for a, dd in out[K].items()}
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--K", type=int, nargs="+", default=[11, 50])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = load_inputs("direction", a.layer, "direction")
    pr = d["role"] == "probe"
    test = np.flatnonzero(d["role"] == "test")
    probe = mf.ProbeReadout(d["X"][pr], d["y"][pr], True)
    near = mf.NearestRealReadout(d["X"][test], d["y"][test])
    pf_vals = np.unique(d["y"][pr]).astype(float)
    pf_C = np.array([d["X"][pr & np.isclose(d["y"], v)].astype(float).mean(0) for v in pf_vals])
    angle = "labels" if a.layer == 22 else "unsupervised"
    out = []
    for s in a.seeds:
        r = run_arc(d, a.layer, s, angle, probe, near, pf_vals, pf_C, a.K)
        r["n_probe_fold_values"] = int(len(pf_vals))
        out.append(r)
        k0 = str(a.K[0])
        print(a.layer, s, r["seconds"], "s", {q: round(float(np.mean(r["K"][k0][q]["radius_min"])), 3)
                                               for q in ARMS}, flush=True)
        Path(a.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()

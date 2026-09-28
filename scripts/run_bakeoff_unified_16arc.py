"""Unified bake-off on the 16 held-out contiguous arcs (same folds, same picked clips, same readers), K = 11 waypoints,
each arm at its own norm and rescaled per clip and per waypoint to the raw chord's full-space ||delta||.

Arms (all additive, residual kept):
  chord_raw, fitpack_smooth, interp_causalab_lam0, causalab_lam_cv   as scripts/run_endpoint_diagnosis_path.py
  fourier2_4d   Z ~ a + B [cos t, sin t, cos 2t, sin 2t] fit by OLS on knot-fold rows at kept values in the same PCA-64
                space (wm.motiongeom.fourier_features / fit_linear_encoding, run_motion_geometry exp3 'fourier2_4d');
                straight edit z + s (F(tgt) - F(src)) B
  probe_qr      Part 1's multi-probe steer (wm.bakeoff.probe_qr_delta), basis refit per arc on knot rows at kept values
                (wm.bakeoff.refit_probe_basis, as run_bakeoff's default); straight edit x + s * delta
Readers as run_endpoint_diagnosis_path.path_metrics (probe radius / angle, nearest-real R).

  python scripts/run_bakeoff_unified_16arc.py --layer 12 --seeds 0 ... --out results/endpoint_diagnosis_raw/unified_L12_a.json
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from wm import bakeoff as bo
from wm import manifold as mf
from wm import motiongeom as mg
from wm.p2_data import load_inputs

_s = importlib.util.spec_from_file_location("edp", Path(__file__).with_name("run_endpoint_diagnosis_path.py"))
edp = importlib.util.module_from_spec(_s)
_s.loader.exec_module(edp)
ed, p2 = edp.ed, edp.p2
K = 11
SCAL = ("radius_min", "radius_mean", "monotone_frac", "ordering_spearman", "R_end_test", "R_wp_mean", "err_end")


def run_arc(d, L, seed, angle, probe, near, pf_vals, pf_C, n_clips=48):
    t0 = time.time()
    m = p2.build(d, 64, angle, "contiguous", seed, 0, "smooth", aim="coord")
    pca, cent, curve, raw_c = m["pca"], m["cent"], m["curve"], m["raw_curve"]
    picks = p2.pick_clips(d, m["held"], n_clips, seed)
    raw_C = {float(v): c for v, c in zip(cent["values"], cent["C"])}
    Ck = np.array([raw_C[float(v)] for v in curve.values])
    f_int, _ = ed.causalab_curve(curve.coords, Ck, 0.0)
    cvl = ed.cv_lambda(np.asarray(curve.coords, float), Ck, curve.values)
    f_cv, _ = ed.causalab_curve(curve.coords, Ck, cvl["lambda_cv"])
    curves = {"fitpack_smooth": curve, "interp_causalab_lam0": edp.FnCurve(f_int, curve),
              "causalab_lam_cv": edp.FnCurve(f_cv, curve)}
    knot = m["knot"]
    Zk = pca.project(d["X"][knot].astype(float))
    _, Bf = mg.fit_linear_encoding(Zk, mg.fourier_features(d["y"][knot].astype(float), 2))
    basis, std, bnote = bo.refit_probe_basis(d, m, "direction")
    s = np.linspace(0.0, 1.0, K)
    acc = {}
    meta = {"src": [], "tgt": [], "clip": []}
    for tgt, pick in picks.items():
        x, src = d["X"][pick].astype(float), d["y"][pick].astype(float)
        Z, resid = pca.project(x), pca.complement(x)
        arc = p2.traversed_arc(curve, src, tgt)
        shift_signed = arc * (((tgt - src) * arc) % 360.0)
        ta, tb = curve.coord_of_value(src), curve.coord_of_value(np.full(len(src), tgt))
        tv = np.full(len(src), tgt)
        Zp = {"chord_raw": mf.linear_coords(Z, mf.piecewise_linear_point(raw_c, src),
                                            mf.piecewise_linear_point(raw_c, tv), K)}
        for a, c in curves.items():
            Zp[a] = mf.manifold_coords(Z, c, ta, tb, K, "shift")
        dF = (mg.fourier_features(tv, 2) - mg.fourier_features(src, 2)) @ Bf
        Zp["fourier2_4d"] = Z[:, None] + s[None, :, None] * dF[:, None]
        W = {a: p2.compose(pca, zp, resid) for a, zp in Zp.items()}
        dq, _ = bo.probe_qr_delta(x, basis, tgt, True, *std)
        W["probe_qr"] = x[:, None] + s[None, :, None] * dq[:, None]
        nl = np.linalg.norm(W["chord_raw"] - x[:, None], axis=-1, keepdims=True)
        for a in list(W):
            if a == "chord_raw":
                continue
            da = W[a] - x[:, None]
            na = np.linalg.norm(da, axis=-1, keepdims=True)
            W[f"{a}@chord_norm"] = x[:, None] + da * np.where(na > 0, nl / np.where(na > 0, na, 1.0), 0.0)
        for a, w in W.items():
            r = edp.path_metrics(w, x, src, tgt, s, shift_signed, probe, near, pf_vals, pf_C)
            r["delta_norm_end"] = np.linalg.norm(w[:, -1] - x, axis=-1)
            e = acc.setdefault(a, {})
            for q in SCAL + ("delta_norm_end",):
                e.setdefault(q, []).append(r[q])
            e.setdefault("radius_wp", []).append(r["radius"])
        meta["src"].append(src); meta["tgt"].append(tv); meta["clip"].append(pick)
    out = {"layer": L, "seed": seed, "angle": angle, "held_out_values": [float(v) for v in m["held"]],
           "lambda_cv": cvl["lambda_cv"], "probe_basis": bnote, "K": K,
           "clip": np.concatenate(meta["clip"]).tolist(), "src": np.concatenate(meta["src"]).tolist(),
           "tgt": np.concatenate(meta["tgt"]).tolist(),
           "arms": {a: {**{q: np.round(np.concatenate(v), 5).tolist() for q, v in e.items() if q != "radius_wp"},
                        "radius_wp_mean": np.concatenate(e["radius_wp"]).mean(0).tolist()} for a, e in acc.items()},
           "seconds": round(time.time() - t0, 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
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
    for sd in a.seeds:
        r = run_arc(d, a.layer, sd, angle, probe, near, pf_vals, pf_C)
        out.append(r)
        print(a.layer, sd, r["seconds"], "s err", {q: round(float(np.mean(r["arms"][q]["err_end"])), 2)
                                                   for q in ("chord_raw", "fourier2_4d", "probe_qr")}, flush=True)
        Path(a.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()

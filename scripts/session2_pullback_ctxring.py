"""Rescore the Goodfire-recipe pullback paths (session2_pullback_goodfire.py) and the old reverse-test paths against
the CONTEXT-ONLY ring: the activation manifold of the activation those runs actually edited.

The pullback edits each carrier's context-only (frames 1-8) point-22 meanpool, but session2_pullback_goodfire.json
scores the path against a spline and chord fitted on FULL-CLIP (16-frame) knot activations; the carriers' own
context-only coordinates sit 0.61-0.69 natural units from their full-clip coordinates, the same size as the reported
residuals. Here:

  rows     CPU, local -> artifacts/session2/ctx_ring/rows.npz: every knot clip (720; the ring uses the 632 at the 56
           kept values, the others give held-value centroids for the scale ratio), the 8 x 16 pullback carriers and the
           20 old reverse-test carriers.
  extract  box, GPU   -> artifacts/session2/ctx_ring/ctx_meanpool_L22.npz (+ .sha256.json): point-22 meanpool of the
           encoder run on frames 1-8 only, the exact code path of session2_pullback_goodfire.run / along_path.reverse
           (preprocess -> prefix(model, pv[:, :8], [22]) -> propagate._mean), batches of 16, fp32.
  score    CPU, local -> results/session2_pullback_goodfire_ctxring.json.

Context-only ring = geometry() of the pullback script with X replaced by the context-only meanpool: PCA-64 fit on
the same 632 knot clips (knot role, values not held out), per-value centroids, interpolating periodic spline
(run_session2 plan spline = interp) on the labels coordinate, and the chord between the source centroid and the
target's spline point (pair_paths).

Path representation. The optimised waypoint V_t (top-32 coordinates of the FULL-CLIP PCA) replaced each carrier's
top-32 full-clip PCA coordinates: edited activation x_n(t) = c_n + (V_t - P32 (c_n - mu_full)) @ C32_full, with c_n
the carrier's context-only meanpool. That D-dim activation is projected into the context-only PCA-64. The shared
path is the carrier mean (the projection is affine, so this is the projection of the mean edited activation);
`per_carrier_mean` repeats the residuals per carrier and averages. Old reverse test: x_n(t) = c_n + w_t @ C64_full
(its edit), per carrier and target, scored against the context-only spline / chord for that carrier's (source,
target) exactly like the pullback (absolute ring positions, unscaled; the original old scoring used carrier-anchored,
natural-norm-scaled deltas).

Units: the same natural-change units as the original file (pullback: mean natural twin change at the same |shift|;
old test: the carrier's own), so every number is directly comparable to session2_pullback_goodfire.json. The
context-only / full-clip ratio of knot-centroid distances (source -> target value) is stored per pair for
conversion.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm.data import PROJECT_ROOT, decode, load_table  # noqa: E402

OUT = PROJECT_ROOT / "artifacts" / "session2"
CR = OUT / "ctx_ring"
PB = OUT / "pullback_goodfire"
RV = OUT / "along_path" / "reverse"
AP = OUT / "along_path"
RES = PROJECT_ROOT / "results"
L, K_OPT = 22, 32


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ================================================================ rows (CPU)
def rows():
    from wm.p2_data import load_inputs
    d = load_inputs("direction", L)
    zb, zr = np.load(PB / "prep.npz"), np.load(RV / "prep.npz")
    n_pairs = len(sorted(PB.glob("pair_*.npz")))
    knot = np.flatnonzero(d["role"] == "knot")
    pb = np.unique(zb["rows"][:n_pairs].ravel())
    old = np.unique(zr["carrier_rows"])
    allr = np.unique(np.concatenate([knot, pb, old]))
    ids = d["df"]["id"].to_numpy()[allr]
    CR.mkdir(parents=True, exist_ok=True)
    np.savez(CR / "rows.npz", rows=allr, ids=ids, knot=knot, pullback_carriers=pb, old_carriers=old)
    print(f"rows: {len(allr)} (knot {len(knot)}, pullback carriers {len(pb)}, old carriers {len(old)})")


# ================================================================ extract (box, GPU)
def extract(batch=16):
    import torch
    from wm.extract import load_model, pick_device, preprocess, set_precision
    from wm.propagate import _mean, prefix
    set_precision()
    device = pick_device()
    model = load_model("vjepa2", device)
    r = np.load(CR / "rows.npz")
    rw = r["rows"]
    df = load_table("direction")
    assert (df["id"].to_numpy()[rw] == r["ids"]).all()
    X = np.zeros((len(rw), 1024), np.float64)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(rw), batch):
            pv = preprocess([decode(df["video"].iloc[int(i)]) for i in rw[s:s + batch]]).to(device)
            saved, _, _ = prefix(model, pv[:, :8], [L])
            X[s:s + batch] = _mean(saved[L]).double().cpu().numpy()
            if s % (batch * 10) == 0:
                print(f"{s + batch}/{len(rw)} {time.time() - t0:.0f}s", flush=True)
    sec = time.time() - t0
    f = CR / "ctx_meanpool_L22.npz"
    np.savez(f, X=X, rows=rw, ids=r["ids"])
    info = {"file": f.name, "sha256": sha256_file(f), "rows_npz_sha256": sha256_file(CR / "rows.npz"),
            "n_clips": int(len(rw)), "point": L, "frames": "1-8 (pv[:, :8])", "batch": batch, "precision": "fp32",
            "seconds_gpu_forward": sec, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    (CR / "ctx_meanpool_L22.sha256.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


# ================================================================ score (CPU)
def ctx_geometry(G, Xc):
    """geometry() of the pullback script on the context-only meanpool: same knot set, k, spline, labels angle."""
    mf, info, knot, y = G["mf"], G["info"], G["knot"], G["y"]
    pca = mf.fit_pca(Xc[knot], info["k"])
    cent = mf.centroids(pca.project(Xc[knot]), y[knot])
    auto = mf.choose_angle_source(cent["C"], cent["values"])
    curve = mf.fit_curve(cent, True, angle="labels", plane="activation", spline=info["spline"])
    return {"pca": pca, "curve": curve, "cent": cent, "auto_angle_choice": {"angle": auto["angle"],
                                                                         "plane": auto["plane"]}}


def ring_paths(curve, src, tg, n, step_sign):
    ta, tb = curve.coord_of_value(src), curve.coord_of_value(tg)
    stp = curve.step(ta, tb)
    stp = np.sign(step_sign) * abs(stp) if abs(abs(stp) - np.pi) < 1e-6 else stp
    s = np.linspace(0, 1, n)
    sp = curve(ta + s * stp)
    return sp, sp[:1] + s[:, None] * (sp[-1:] - sp[:1]), float(stp)


def dist_to_ring(Z, curve, n=2000):
    """Closest-point distance of each row of Z to the closed dense ring."""
    import session2_pullback_goodfire as pbg
    P = curve.dense(n)[1]
    P = np.vstack([P, P[:1]])
    Z = np.atleast_2d(Z)
    return np.linalg.norm(Z - pbg.project_to_polyline(Z, P), axis=-1)


def score():
    import session2_pullback_goodfire as pbg
    from run_session2 import write
    G = pbg.geometry()
    y, knot, curve_f, pca_f, mf, d = G["y"], G["knot"], G["curve"], G["pca"], G["mf"], G["d"]
    zc = np.load(CR / "ctx_meanpool_L22.npz")
    ext = json.loads((CR / "ctx_meanpool_L22.sha256.json").read_text())
    assert sha256_file(CR / "ctx_meanpool_L22.npz") == ext["sha256"]
    Xc = np.full_like(G["X"], np.nan)
    Xc[zc["rows"]] = zc["X"]
    assert np.isfinite(Xc[knot]).all()
    C = ctx_geometry(G, Xc)
    pca_c, curve_c = C["pca"], C["curve"]
    z = np.load(PB / "prep.npz")
    C32f, muf = z["components"][:K_OPT], z["pca_mean"]
    assert np.allclose(C32f, pca_f.components[:K_OPT]) and np.allclose(muf, pca_f.mean)
    files = sorted(PB.glob("pair_*.npz"))
    kn_all = d["role"] == "knot"

    def cdist(X, a, b):
        return float(np.linalg.norm(X[kn_all & (y == a)].mean(0) - X[kn_all & (y == b)].mean(0)))

    per, per32, perc, parity = [], [], [], []
    for f in files:
        i = int(f.stem.split("_")[1])
        r = np.load(f)
        s, tg, ss, unit = float(z["src"][i]), float(z["tg"][i]), float(z["shift_signed"][i]), float(z["nat_unit"][i])
        rows_i = z["rows"][i]
        c = Xc[rows_i]                                                                  # [16, D] context-only
        zc64_here = (c - muf) @ z["components"].T
        parity.append(float(np.abs(zc64_here - r["zc64"]).max() / np.abs(r["zc64"]).max()))
        Xe = c[None] + (r["V"][:, None, :] - zc64_here[None, :, :K_OPT]) @ C32f          # [K, 16, D] edited
        Ze = pca_c.project(Xe.reshape(-1, Xe.shape[-1])).reshape(Xe.shape[0], Xe.shape[1], -1)
        v = Ze.mean(1)                                                                  # shared path, ctx PCA-64
        sp_f, _ = pbg.pair_paths(G, s, tg, pbg.K)
        step_f = curve_f.step(curve_f.coord_of_value(s), curve_f.coord_of_value(tg))
        sp, ch, step_c = ring_paths(curve_c, s, tg, pbg.K, step_f)
        same_dir = bool(np.sign(step_c) == np.sign(step_f))
        sc = pbg.score_triplet(v, sp, ch, unit)
        sc32 = pbg.score_triplet(v[:, :K_OPT], sp[:, :K_OPT], ch[:, :K_OPT], unit)
        pc = [pbg.score_triplet(Ze[:, n], sp, ch, unit) for n in range(Ze.shape[1])]
        zcar = pca_c.project(c)
        zcar_f = pca_f.project(c)                                                       # ctx carriers, full-clip PCA
        zfull_f = pca_f.project(G["X"][rows_i])                                         # full-clip carriers
        rec = {"pair": f"{s}->{tg}", "shift_signed_deg": ss, "nat_unit": unit, "ctx_step_same_direction": same_dir,
               **sc,
               "carrier_offset_from_ctx_ring_over_unit": {
                   "mean_carrier_to_ring_closest": float(dist_to_ring(zcar.mean(0), curve_c)[0] / unit),
                   "per_carrier_to_ring_closest_mean": float(dist_to_ring(zcar, curve_c).mean() / unit),
                   "mean_carrier_to_source_centroid": float(np.linalg.norm(zcar.mean(0) - sp[0]) / unit),
                   "mean_carrier_to_source_centroid_pca32": float(np.linalg.norm(zcar.mean(0)[:K_OPT] - sp[0, :K_OPT])
                                                                  / unit)},
               "carrier_offset_from_fullclip_ring_over_unit (ctx carriers, as the original scored)": {
                   "mean_carrier_to_ring_closest": float(dist_to_ring(zcar_f.mean(0), curve_f)[0] / unit),
                   "per_carrier_to_ring_closest_mean": float(dist_to_ring(zcar_f, curve_f).mean() / unit),
                   "mean_carrier_to_source_centroid": float(np.linalg.norm(zcar_f.mean(0) - sp_f[0]) / unit),
                   "mean_carrier_to_source_centroid_pca32": float(np.linalg.norm(zcar_f.mean(0)[:K_OPT]
                                                                                 - sp_f[0, :K_OPT]) / unit)},
               "carrier_ctx_vs_fullclip_pca32_offset_over_unit (recomputed)": float(np.linalg.norm(
                   zcar_f[:, :K_OPT] - zfull_f[:, :K_OPT], axis=-1).mean() / unit),
               "ctx_over_full_knot_centroid_distance_src_to_tg": cdist(Xc, s, tg) / cdist(G["X"], s, tg),
               "per_carrier_mean": {k: float(np.mean([p[k] for p in pc])) for k in
                                    ("resid_to_spline", "resid_to_chord", "r2_spline", "r2_chord")}}
        per.append(rec)
        per32.append(sc32)
        perc.append({k: rec["per_carrier_mean"][k] for k in rec["per_carrier_mean"]}
                    )

    # old reverse test, same scoring
    zr, zp, zo = np.load(RV / "prep.npz"), np.load(AP / "plan.npz"), np.load(RV / "reverse_out.npz")
    C64f = zr["components"]
    assert np.allclose(C64f, pca_f.components)                                     # old edits live in the same basis
    old = []
    for n in range(len(zr["pick"])):
        row = int(zr["carrier_rows"][n])
        cn = Xc[row]
        rec = []
        for j in range(2):
            cc, tt = zr["pick"][n], zr["tj"][n, j]
            s, tg = float(y[row]), float(zp["targets"][cc, tt])
            Ze = pca_c.project(cn[None] + zo["w"][n, j] @ C64f)                        # [8, 64]
            step_sign = float(zr["shift_signed"][n, j])
            sp, ch, _ = ring_paths(curve_c, s, tg, 41, step_sign)
            rec.append(pbg.score_triplet(Ze, sp, ch, float(zr["nat"][n, j])))
        old.append({k: float(np.mean([q[k] for q in rec])) for k in rec[0] if not k.endswith("_by_t")}
                   | {"resid_to_spline_by_t": np.mean([q["resid_to_spline_by_t"] for q in rec], 0).tolist(),
                      "resid_to_chord_by_t": np.mean([q["resid_to_chord_by_t"] for q in rec], 0).tolist()})

    role = d["role"]
    fill = [int((role[z["rows"][int(f.stem.split("_")[1])]] == "knot").sum()) for f in files]
    n_test = [int((role[z["rows"][int(f.stem.split("_")[1])]] == "test").sum()) for f in files]
    data_sep = {
        "carriers": f"each pair's 16 carriers = every probe/test clip at the source value, filled to 16 with seeded "
        f"knot clips: knot fill per pair {fill} (of 16); test clips per pair {n_test}. The knot clips are also in the "
        "PCA / ring fit set (full-clip and context-only).",
        "kappa": "the von Mises readout concentration kappa was fitted by max likelihood on the TEST clips' unedited "
        "forecasts, and test clips are also carriers",
        "behaviour_centroids": "the behaviour centroids b_v (and so the target spline) used ALL clips at each value "
        "(knot + probe + test, carriers included), unedited forecasts",
        "readout": "the native direction probe (W, b) was fit on probe clips; probe clips are also carriers"}
    orig = json.loads((RES / "session2_pullback_goodfire.json").read_text())
    ogf = orig["goodfire"]
    off = lambda k: pbg.se([p[k]["mean_carrier_to_ring_closest"] for p in per])  # noqa: E731
    out = {"point": L, "K": pbg.K, "k_opt": K_OPT, "n_carriers_per_pair": pbg.N_CAR, "n_pairs": len(per),
           "goodfire": {"per_pair": per, "summary_64d": pbg.summarise(per), "summary_32d": pbg.summarise(per32),
                        "summary_64d_per_carrier_mean": {k: pbg.se([p[k] for p in perc]) for k in
                                                         ("resid_to_spline", "resid_to_chord", "r2_spline",
                                                          "r2_chord")}
                        | {"paired_resid_spline_minus_chord": pbg.paired([p["resid_to_spline"] for p in perc],
                                                                         [p["resid_to_chord"] for p in perc])},
                        "carrier_offset_from_ctx_ring_over_unit": {
                            "mean_carrier_to_ring_closest": off("carrier_offset_from_ctx_ring_over_unit"),
                            "mean_carrier_to_source_centroid": pbg.se(
                                [p["carrier_offset_from_ctx_ring_over_unit"]["mean_carrier_to_source_centroid"]
                                 for p in per])},
                        "carrier_offset_from_fullclip_ring_over_unit": {
                            "mean_carrier_to_ring_closest": off(
                                "carrier_offset_from_fullclip_ring_over_unit (ctx carriers, as the original scored)"),
                            "mean_carrier_to_source_centroid": pbg.se(
                                [p["carrier_offset_from_fullclip_ring_over_unit (ctx carriers, as the original "
                                   "scored)"]["mean_carrier_to_source_centroid"] for p in per])},
                        "original_fullclip_summary_64d_means": {k: ogf["summary_64d"]["means"][k] for k in
                                                                ("resid_to_spline", "resid_to_chord", "r2_spline",
                                                                 "r2_chord", "chord_baseline_resid_to_spline")},
                        "original_fullclip_paired_resid_spline_minus_chord": ogf["summary_64d"][
                            "paired_resid_spline_minus_chord"]},
           "old_reverse_test": {"n_carriers": len(old), "per_carrier": old, "summary": pbg.summarise(old),
                                "original_fullclip_summary_means": {
                                    k: orig["old_reverse_test"]["summary"]["means"][k] for k in
                                    ("resid_to_spline", "resid_to_chord", "r2_spline", "r2_chord",
                                     "chord_baseline_resid_to_spline")}},
           "data_separation": data_sep,
           "ctx_ring_provenance": {
               "activations": {**ext, "path": str((CR / "ctx_meanpool_L22.npz").relative_to(PROJECT_ROOT))},
               "context_definition": "frames 1-8 of the 16-frame clip (pv[:, :8]), encoder prefix to point 22, "
               "propagate._mean over the 1024 context tokens; the code path of session2_pullback_goodfire.run",
               "parity_vs_stored_carrier_zc64_rel_maxabs": float(max(parity)),
               "pca_fit_set": f"{int(knot.sum())} knot clips at the {len(C['cent']['values'])} kept values "
               "(held-out values excluded), PCA-64 (as geometry())",
               "spline": G["info"]["spline"], "angle": "labels (forced, as the full-clip ring's choice at point 22)",
               "auto_angle_choice_on_ctx_centroids": C["auto_angle_choice"],
               "ctx_pca_explained_top32_over_top64": float(pca_c.explained[:K_OPT].sum() / pca_c.explained.sum()),
               "path_representation": "edited activation per carrier (context-only meanpool with its top-32 "
               "full-clip PCA coordinates replaced by the optimised waypoint) projected into the context-only "
               "PCA-64; shared path = carrier mean. Old test: carrier context-only meanpool + w @ full-clip C64.",
               "units": "natural-change units of the original file (unchanged)",
               "summary_32d": "truncation to the top-32 CONTEXT-ONLY PCs",
               "original_file": "results/session2_pullback_goodfire.json",
               "original_file_sha256": sha256_file(RES / "session2_pullback_goodfire.json"),
               "pair_npz_sha256": {f.name: sha256_file(f) for f in files},
               "reverse_out_sha256": sha256_file(RV / "reverse_out.npz")}}
    write(RES / "session2_pullback_goodfire_ctxring.json", out, stage="pullback_goodfire_ctxring_score",
          seeds=None, cache=str(CR))
    for nm, sm in (("ctx64", out["goodfire"]["summary_64d"]), ("ctx32", out["goodfire"]["summary_32d"]),
                   ("old", out["old_reverse_test"]["summary"])):
        print(nm, {k: round(v["mean"], 3) for k, v in sm["means"].items()},
              "paired", {k: sm["paired_resid_spline_minus_chord"][k] for k in ("mean_diff", "ci95_t", "p_wilcoxon")})
    print("parity", max(parity), "offset ctx", out["goodfire"]["carrier_offset_from_ctx_ring_over_unit"])


if __name__ == "__main__":
    a = sys.argv[1]
    {"rows": rows, "extract": extract, "score": score}[a]()

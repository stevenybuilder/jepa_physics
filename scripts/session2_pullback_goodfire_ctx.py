"""Goodfire-recipe pullback (session2_pullback_goodfire.py) rerun with ALL geometry from CONTEXT-ONLY activations.

The first run edited each carrier's context-only (frames 1-8) point-22 meanpool but took its PCA-64 basis, the replaced
top-32 components, the source / target centroids and the chord initialisation from FULL-CLIP knot activations, so its
path started ~0.57 natural units off the context-only ring and the ctx-ring rescoring (session2_pullback_goodfire_ctxring
.py) could change only the ruler. Here every geometric object is context-only:

  PCA-64      fit on the 632 context-only knot meanpools at the 56 kept values (artifacts/session2/ctx_ring/
              ctx_meanpool_L22.npz, sha256-checked) = ctxring.ctx_geometry
  edit        replacement of the carriers' top-32 coordinates in THAT basis (run() of the pullback script, unchanged,
              reading components / pca_mean / init32 from this directory's prep.npz)
  ring        per-value context-only centroids, interpolating periodic spline on the labels coordinate
  chord       straight line from the context-only source centroid to the target's context-only spline point
  init        that chord's top-32 coordinates at the 20 t's

Unchanged from the first run: the 8 (source, target) pairs, the 16 carriers per pair, the behaviour target (squared
Hellinger to the von Mises-readout behaviour-manifold spline, same kappa and centroids), L-BFGS (lr 1, strong Wolfe,
max_iter 5, tol 1e-3 over 2 steps), no norm term, the 32-evaluation cap per pair, natural-change units. The arc
direction of the ctx spline is forced to the behaviour target's signed shift.

Scoring (A.9, pbg.score_triplet) against the context-only spline and chord. Path representation = the carrier-mean
edited activation projected into the context-only PCA-64 (top-32 = the waypoint exactly, PCs 33-64 = the carriers'
own mean); the causalab zero-padded path and a top-32 truncation are stored too.

  prep   CPU, local  -> artifacts/session2/pullback_goodfire_ctx/prep.npz
  run    box, GPU    -> python scripts/session2_pullback_goodfire.py run 8 32 16 0 artifacts/session2/pullback_goodfire_ctx
  score  CPU, local  -> results/session2_pullback_goodfire_ctx.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm.data import PROJECT_ROOT  # noqa: E402

import session2_pullback_ctxring as ctr  # noqa: E402
import session2_pullback_goodfire as pbg  # noqa: E402

PB, CR, PBC = pbg.PB, ctr.CR, pbg.OUT / "pullback_goodfire_ctx"
RES, K, K_OPT = pbg.RES, pbg.K, pbg.K_OPT


def ctx_setup():
    G = pbg.geometry()
    ext = json.loads((CR / "ctx_meanpool_L22.sha256.json").read_text())
    assert ctr.sha256_file(CR / "ctx_meanpool_L22.npz") == ext["sha256"]
    zc = np.load(CR / "ctx_meanpool_L22.npz")
    Xc = np.full_like(G["X"], np.nan)
    Xc[zc["rows"]] = zc["X"]
    assert np.isfinite(Xc[G["knot"]]).all() and int(G["knot"].sum()) == 632
    C = ctr.ctx_geometry(G, Xc)
    return G, Xc, C, ext


def ctx_ring_path(curve, mf, src, tg, shift_signed, n):
    """Context-only spline from the source centroid to the target's spline point, going the way (in theta) the
    behaviour target goes (sign of shift_signed), and the chord between its ends."""
    ta, tb = curve.coord_of_value(src), curve.coord_of_value(tg)
    stp = curve.step(ta, tb)
    orient = np.sign(mf.wrap_pi(curve.coord_of_value(src + 1.0) - ta))
    flipped = False
    if np.sign(stp) * orient != np.sign(shift_signed):
        assert abs(abs(pbg.signed_wrap(tg - src)) - 180) < 1e-6, (src, tg)   # only a 180-degree pair may flip
        stp, flipped = stp - np.sign(stp) * 2 * np.pi, True
    s = np.linspace(0, 1, n)
    sp = curve(ta + s * stp)
    return sp, sp[:1] + s[:, None] * (sp[-1:] - sp[:1]), float(stp), flipped


def prep():
    G, Xc, C, _ = ctx_setup()
    pca_c, curve_c, mf = C["pca"], C["curve"], G["mf"]
    probe = Xc[G["knot"]][:5]
    assert np.allclose(pca_c.project(probe), (probe - pca_c.mean) @ pca_c.components.T)
    z = np.load(PB / "prep.npz")
    sp_l, ch_l, flips = [], [], []
    for i in range(len(z["src"])):
        sp, ch, _, fl = ctx_ring_path(curve_c, mf, float(z["src"][i]), float(z["tg"][i]), float(z["shift_signed"][i]),
                                      K)
        sp_l.append(sp)
        ch_l.append(ch)
        flips.append(fl)
    sp_a, ch_a = np.array(sp_l), np.array(ch_l)
    keep = {k: z[k] for k in ("src", "tg", "shift_signed", "rows", "target_p", "nat_unit", "unedited_loss_per_t", "W",
                              "b", "U", "kappa", "ts", "values", "behaviour_centroids")}
    PBC.mkdir(parents=True, exist_ok=True)
    np.savez(PBC / "prep.npz", **keep, spline64=sp_a, chord64=ch_a, init32=ch_a[:, :, :K_OPT],
             pca_mean=pca_c.mean, components=pca_c.components,
             zctx_carriers=np.array([pca_c.project(Xc[r]) for r in z["rows"]]))
    meta = {"geometry": "context-only (frames 1-8) point-22 meanpool; PCA-64 on the 632 knot clips at kept values",
            "n_pairs": int(len(z["src"])), "arc_flipped_to_behaviour_direction": flips,
            "source_prep_sha256": ctr.sha256_file(PB / "prep.npz")}
    (PBC / "prep.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def _offsets(zcar, sp, curve, unit):
    return {"mean_carrier_to_ring_closest": float(ctr.dist_to_ring(zcar.mean(0), curve)[0] / unit),
            "per_carrier_to_ring_closest_mean": float(ctr.dist_to_ring(zcar, curve).mean() / unit),
            "mean_carrier_to_source_centroid": float(np.linalg.norm(zcar.mean(0) - sp[0]) / unit),
            "mean_carrier_to_source_centroid_pca32": float(np.linalg.norm(zcar.mean(0)[:K_OPT] - sp[0, :K_OPT]) / unit)}


def score():
    from run_session2 import write
    G, Xc, C, ext = ctx_setup()
    pca_c, curve_c, mf, y, d = C["pca"], C["curve"], G["mf"], G["y"], G["d"]
    z = np.load(PBC / "prep.npz")
    assert np.allclose(z["components"], pca_c.components) and np.allclose(z["pca_mean"], pca_c.mean)
    C32, C64, mu = pca_c.components[:K_OPT], pca_c.components, pca_c.mean
    U, kappa, W, b = z["U"], float(z["kappa"]), z["W"], z["b"]
    files = sorted(PBC.glob("pair_*.npz"))
    keys4 = ("resid_to_spline", "resid_to_chord", "r2_spline", "r2_chord")
    per, per32, perpad, perc, parity = [], [], [], [], []
    for f in files:
        i = int(f.stem.split("_")[1])
        r = np.load(f)
        s, tg, ss, unit = float(z["src"][i]), float(z["tg"][i]), float(z["shift_signed"][i]), float(z["nat_unit"][i])
        c = Xc[z["rows"][i]]                                                             # [16, D] Mac ctx meanpool
        zc64 = (c - mu) @ C64.T
        parity.append(float(np.abs(zc64 - r["zc64"]).max() / np.abs(r["zc64"]).max()))
        sp, ch = z["spline64"][i], z["chord64"][i]

        def edited(Vp):
            Xe = c[None] + (Vp[:, None, :] - zc64[None, :, :K_OPT]) @ C32                # [K, 16, D]
            return pca_c.project(Xe.reshape(-1, Xe.shape[-1])).reshape(Xe.shape[0], Xe.shape[1], -1)

        Ze, Zi = edited(r["V"]), edited(z["init32"][i])
        assert np.allclose(Ze[..., :K_OPT], r["V"][:, None].repeat(Ze.shape[1], 1), atol=1e-6)
        v = Ze.mean(1)
        sc = pbg.score_triplet(v, sp, ch, unit)
        si = pbg.score_triplet(Zi.mean(1), sp, ch, unit)
        sc32 = pbg.score_triplet(r["V"], sp[:, :K_OPT], ch[:, :K_OPT], unit)
        scpad = pbg.score_triplet(np.hstack([r["V"], np.zeros((K, 64 - K_OPT))]), sp, ch, unit)
        pc = [pbg.score_triplet(Ze[:, n], sp, ch, unit) for n in range(Ze.shape[1])]
        tp = z["target_p"][i]
        lf = {}
        for nm in ("final", "init"):
            pdist = pbg.readout_dist(r[f"pred_{nm}"].astype(np.float64).mean(-2) @ W + b, U, kappa)
            lf[nm] = float((1 - np.sqrt(pdist * tp[:, None]).sum(-1)).mean(1).sum())
        Pf = r["pred_final"].astype(np.float64).mean(-2) @ W + b
        ang = np.degrees(np.arctan2(Pf[..., 0], Pf[..., 1])) % 360.0
        err = np.abs(pbg.signed_wrap(ang - ((s + z["ts"] * ss) % 360.0)[:, None])).mean(1)
        rec = {"pair": f"{s}->{tg}", "shift_signed_deg": ss, "nat_unit": unit, **sc,
               "pathmean_resid_to_spline": sc["resid_to_spline"], "pathmean_resid_to_chord": sc["resid_to_chord"],
               "start_resid_to_spline_t0": sc["resid_to_spline_by_t"][0],
               "end_resid_to_spline_t1": sc["resid_to_spline_by_t"][-1],
               "start_resid_to_chord_t0": sc["resid_to_chord_by_t"][0],
               "end_resid_to_chord_t1": sc["resid_to_chord_by_t"][-1],
               "chord_own_start_resid_to_spline_t0": float(pbg.recap_metrics(ch, sp)["residual_by_t"][0] / unit),
               "chord_own_end_resid_to_spline_t1": float(pbg.recap_metrics(ch, sp)["residual_by_t"][-1] / unit),
               "chord_own_pathmean_resid_to_spline": sc["chord_baseline_resid_to_spline"],
               "carrier_offset_from_ctx_ring_over_unit": _offsets(pca_c.project(c), sp, curve_c, unit),
               "init_ctx_chord_path": {k: si[k] for k in keys4 + ("resid_to_spline_by_t", "resid_to_chord_by_t")}
               | {"pathmean_resid_to_spline": si["resid_to_spline"], "start_resid_to_spline_t0":
                  si["resid_to_spline_by_t"][0], "end_resid_to_spline_t1": si["resid_to_spline_by_t"][-1]},
               "padded_path_goodfire": {k: scpad[k] for k in keys4},
               "per_carrier_mean": {k: float(np.mean([p[k] for p in pc])) for k in keys4},
               "loss_init_bf16": float(r["eval_losses"][0]), "loss_final_bf16": float(r["best_loss"]),
               "loss_final_fp32": lf["final"], "loss_chord_init_fp32": lf["init"],
               "loss_unedited": float(z["unedited_loss_per_t"][i].sum()),
               "loss_history": r["history"].round(5).tolist(), "eval_losses": r["eval_losses"].round(5).tolist(),
               "outer_steps": int(r["outer_steps"]), "n_evals": int(r["n_evals"]), "stop": str(r["stop"]),
               "optimiser": str(r["optimiser"]), "seconds": float(r["seconds"]),
               "forecast_err_to_ideal_deg_by_t": err.round(2).tolist(),
               "path_move_over_unit": float(np.linalg.norm(r["V"] - z["init32"][i], axis=-1).mean() / unit)}
        per.append(rec)
        per32.append(sc32)
        perpad.append(scpad)
        perc.append(rec["per_carrier_mean"])

    # old reverse test (session2_reverse_path.json paths), against the context-only ring; + zero-edit baseline
    zr, zp, zo = np.load(ctr.RV / "prep.npz"), np.load(ctr.AP / "plan.npz"), np.load(ctr.RV / "reverse_out.npz")
    C64f = zr["components"]
    assert np.allclose(C64f, G["pca"].components)
    old, zero = [], []
    for n in range(len(zr["pick"])):
        row = int(zr["carrier_rows"][n])
        cn = Xc[row]
        ro, rz = [], []
        for j in range(2):
            s, tg = float(y[row]), float(zp["targets"][zr["pick"][n], zr["tj"][n, j]])
            sp, ch, _ = ctr.ring_paths(curve_c, s, tg, 41, float(zr["shift_signed"][n, j]))
            unit = float(zr["nat"][n, j])
            ro.append(pbg.score_triplet(pca_c.project(cn[None] + zo["w"][n, j] @ C64f), sp, ch, unit))
            z0 = np.repeat(pca_c.project(cn[None]), zo["w"].shape[2], 0)                 # zero edit, same 8 t's
            rz.append({"resid_to_spline": pbg.recap_metrics(z0, sp)["mean_closest_point_residual"] / unit,
                       "resid_to_chord": pbg.recap_metrics(z0, ch, basis_ref=sp)["mean_closest_point_residual"] / unit,
                       "to_source_centroid": float(np.linalg.norm(z0[0] - sp[0]) / unit)})
        old.append({k: float(np.mean([q[k] for q in ro])) for k in ro[0] if not k.endswith("_by_t")}
                   | {"resid_to_spline_by_t": np.mean([q["resid_to_spline_by_t"] for q in ro], 0).tolist(),
                      "resid_to_chord_by_t": np.mean([q["resid_to_chord_by_t"] for q in ro], 0).tolist()})
        zero.append({k: float(np.mean([q[k] for q in rz])) for k in rz[0]} | {"carrier_row": row})
    zsum = {k: pbg.se([q[k] for q in zero]) for k in ("resid_to_spline", "resid_to_chord", "to_source_centroid")}
    zsum["paired_old_path_minus_zero_edit_resid_to_spline"] = pbg.paired([o["resid_to_spline"] for o in old],
                                                                         [q["resid_to_spline"] for q in zero])
    zsum["paired_old_path_minus_zero_edit_resid_to_chord"] = pbg.paired([o["resid_to_chord"] for o in old],
                                                                        [q["resid_to_chord"] for q in zero])

    cr = json.loads((RES / "session2_pullback_goodfire_ctxring.json").read_text())
    sm = lambda L_, k: pbg.se([p[k] for p in L_])  # noqa: E731
    run_info = json.loads((PBC / "run_info.json").read_text()) if (PBC / "run_info.json").exists() else None
    out = {"point": pbg.L, "K": K, "k_opt": K_OPT, "n_carriers_per_pair": pbg.N_CAR, "n_pairs": len(per),
           "run": run_info,
           "goodfire": {
               "per_pair": per, "summary_64d": pbg.summarise(per), "summary_32d": pbg.summarise(per32),
               "summary_64d_padded_goodfire": pbg.summarise(perpad),
               "summary_64d_per_carrier_mean": {k: pbg.se([p[k] for p in perc]) for k in keys4}
               | {"paired_resid_spline_minus_chord": pbg.paired([p["resid_to_spline"] for p in perc],
                                                                [p["resid_to_chord"] for p in perc])},
               "init_ctx_chord_path": {k: pbg.se([p["init_ctx_chord_path"][k] for p in per]) for k in keys4},
               "paired_end_minus_start_resid_to_spline": pbg.paired(
                   [p["resid_to_spline"] for p in per], [p["init_ctx_chord_path"]["resid_to_spline"] for p in per]),
               "path_residual_by_position (optimised path, over pairs)": {k: sm(per, k) for k in (
                   "start_resid_to_spline_t0", "pathmean_resid_to_spline", "end_resid_to_spline_t1",
                   "start_resid_to_chord_t0", "pathmean_resid_to_chord", "end_resid_to_chord_t1")},
               "init_ctx_chord_path_by_position": {k: pbg.se([p["init_ctx_chord_path"][k] for p in per]) for k in (
                   "start_resid_to_spline_t0", "pathmean_resid_to_spline", "end_resid_to_spline_t1")},
               "chord_own_resid_to_spline_by_position": {k: sm(per, k) for k in (
                   "chord_own_start_resid_to_spline_t0", "chord_own_pathmean_resid_to_spline",
                   "chord_own_end_resid_to_spline_t1")},
               "path_move_over_unit": sm(per, "path_move_over_unit"),
               "field_naming": "resid_to_* and pathmean_* = mean over the 20 waypoints; *_t0 = the t = 0 waypoint; "
               "*_t1 = the t = 1 waypoint; carrier offsets = the unedited carriers (the start state before any edit)",
               "carrier_offset_from_ctx_ring_over_unit": {
                   k: pbg.se([p["carrier_offset_from_ctx_ring_over_unit"][k] for p in per])
                   for k in ("mean_carrier_to_ring_closest", "per_carrier_to_ring_closest_mean",
                             "mean_carrier_to_source_centroid")},
               "losses": {k: sm(per, k) for k in ("loss_unedited", "loss_chord_init_fp32", "loss_final_fp32",
                                                  "loss_final_bf16")},
               "paired_loss_final_minus_chord_init": pbg.paired([p["loss_final_fp32"] for p in per],
                                                                [p["loss_chord_init_fp32"] for p in per]),
               "forecast_err_to_ideal_deg_mean_over_t": pbg.se([np.mean(p["forecast_err_to_ideal_deg_by_t"])
                                                                for p in per]),
               "previous_run_ctxring_summary_64d_means": {k: cr["goodfire"]["summary_64d"]["means"][k] for k in
                                                         ("resid_to_spline", "resid_to_chord", "r2_spline", "r2_chord",
                                                          "chord_baseline_resid_to_spline")},
               "previous_run_ctxring_init_fullclip_chord_path": cr["goodfire"]["init_fullclip_chord_path"]},
           "old_reverse_test": {"n_carriers": len(old), "per_carrier": old, "summary": pbg.summarise(old),
                                "zero_edit_baseline": {"per_carrier": zero, "summary": zsum,
                                                       "definition": "each carrier's UNEDITED context-only meanpool "
                                                       "(projected into the context-only PCA-64), held at all 8 "
                                                       "waypoints, scored against the same context-only spline / "
                                                       "chord (source -> target, both targets averaged) in the "
                                                       "carrier's own natural units"}},
           "data_separation": cr["data_separation"],
           "ctx_geometry_provenance": {
               "activations": {**ext, "path": str((CR / "ctx_meanpool_L22.npz").relative_to(PROJECT_ROOT))},
               "pca_fit_set": "632 context-only knot meanpools at the 56 kept values (held-out values excluded)",
               "replaced_basis": "top-32 components of the context-only PCA-64",
               "spline": G["info"]["spline"], "angle": "labels (forced)",
               "auto_angle_choice_on_ctx_centroids": C["auto_angle_choice"],
               "init": "chord from the context-only source centroid to the target's context-only spline point, "
               "top-32 coordinates at the 20 t's",
               "arc_direction": "ctx spline forced to the behaviour target's signed shift "
               f"(flipped pairs: {json.loads((PBC / 'prep.json').read_text())['arc_flipped_to_behaviour_direction']})",
               "parity_box_zc64_vs_mac_ctx_meanpool_rel_maxabs": float(max(parity)),
               "path_representation": "carrier-mean edited activation in the context-only PCA-64 (top-32 = the "
               "waypoint, PCs 33-64 = the carriers' own); summary_64d_padded_goodfire = causalab zero padding; "
               "summary_32d = top-32 truncation",
               "units": "natural-change units of session2_pullback_goodfire.json (unchanged)",
               "prep_npz_sha256": ctr.sha256_file(PBC / "prep.npz"),
               "pair_npz_sha256": {f.name: ctr.sha256_file(f) for f in files},
               "reverse_out_sha256": ctr.sha256_file(ctr.RV / "reverse_out.npz")}}
    write(RES / "session2_pullback_goodfire_ctx.json", out, stage="pullback_goodfire_ctx_score", seeds={"carriers": 0},
          cache=str(PBC))
    g = out["goodfire"]
    for p in per:
        print(f"{p['pair']:>18} start {p['init_ctx_chord_path']['resid_to_spline']:.3f} end_sp {p['resid_to_spline']:.3f}"
              f" end_ch {p['resid_to_chord']:.3f} r2sp {p['r2_spline']:.3f} r2ch {p['r2_chord']:.3f}"
              f" loss {p['loss_final_fp32']:.3f} evals {p['n_evals']}")
    print("means", {k: round(v["mean"], 3) for k, v in g["summary_64d"]["means"].items()})
    print("paired", g["summary_64d"]["paired_resid_spline_minus_chord"])
    print("init", g["init_ctx_chord_path"], "offset", g["carrier_offset_from_ctx_ring_over_unit"])
    print("old", out["old_reverse_test"]["summary"]["means"]["resid_to_spline"], "zero", zsum["resid_to_spline"],
          "parity", max(parity))


if __name__ == "__main__":
    {"prep": prep, "score": score}[sys.argv[1]]()

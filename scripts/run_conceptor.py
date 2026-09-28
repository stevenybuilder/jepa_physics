"""Conceptor steering arms (COAST, arXiv 2605.17144) beside the Part 2 spline and raw-chord arms, direction set.

Reuses run_part2.build / pick_clips / evaluate with the headline run's configuration (contiguous held-out block,
smoothing spline, PCA-64, K = 50, n_clips = 48, --seed picks the block and the clips), so the spline (manifold) and
raw-chord (linear_raw) rows are recomputed on the same clips and checked against the stored run.
Conceptors live in the same PCA-64 coordinates; every arm keeps each clip's off-subspace residual.
  coast_a            COAST gate u' = u + beta (C - I) u on the UNCENTRED in-subspace coordinates u = V x (COAST gates
                     the raw residual h), beta 0 -> 1 over the K waypoints; coast_a_b0.1 / _b0.3 stop at COAST's
                     retained betas; coast_a_centered gates the PCA-centred coordinates instead (sensitivity)
  coast_b            target-aimed: z' = z + s C (mu_target - z), mu_target = the raw chord's point at the target
  random_a/random_b  the same two updates with C replaced by a random orthogonal projector of rank round(tr C)
C = C_target AND NOT C_source, both fit on knot clips (train folds 0-2, kept values only) in PCA-64 coordinates.
Target condition = the knot clips of the kept values the raw chord interpolates between at the target, weighted by
the chord's interpolation weights; source condition = the same rule at the source value (one knot value when kept).
Aperture: COAST Stage 2 (mean overlap over all (kept value, target) pairs, band [0.85, 0.95], else closest).

  python scripts/run_conceptor.py --layer 12
  python scripts/run_conceptor.py --layer 12 --seed 3 --out results/arcs_conceptor/L12_s3/p2_conceptor_direction_L12.json --no-fig
  python scripts/run_conceptor.py --aggregate 12
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_part2 as rp                                                   # noqa: E402
from wm import conceptor as cc                                          # noqa: E402
from wm import geometry_checks as gc                                    # noqa: E402
from wm import manifold as mf                                           # noqa: E402
from wm.bakeoff import effective_rank                                   # noqa: E402
from wm.data import PROJECT_ROOT                                        # noqa: E402
from wm.p2_data import load_inputs                                      # noqa: E402
from wm.provenance import provenance                                    # noqa: E402

OFF = ("d_speed", "d_pos", "ring_frac")
KEEP = ("probe_err_to_target", "nearest_real_R", "probe_radius_min", "ordering_spearman", "probe_err_path",
        "delta_norm", "intermediate_mass", "excess_to_nearest_real")
LABELS22_SEEDS = (4, 8, 9, 10)          # point-22 arcs read from their --labels-angle reruns (REPORT section 4.3)


def cond_conceptor(Zk, yk, kv, w, alpha, mode):
    Z, ws = cc.condition_states(Zk, yk, kv, w)
    return cc.conceptor(Z, alpha, ws)


def stats(v, ids, seed=0):
    return rp.paired_bootstrap(v, ids, seed=seed)


def run(args):
    L = args.layer
    angle = "labels" if args.labels_angle else "unsupervised"
    d = load_inputs("direction", L)
    m = rp.build(d, 64, angle, "contiguous", args.seed, 0, "smooth", aim=args.aim)
    pca, raw, periodic = m["pca"], m["raw_curve"], True
    probe_rows = d["role"] == "probe"
    test = np.flatnonzero(d["role"] == "test")
    ctx = {"periodic": periodic, "probe": mf.ProbeReadout(d["X"][probe_rows], d["y"][probe_rows], periodic),
           "near": mf.NearestRealReadout(d["X"][test], d["y"][test]), "near_ctx": None}
    ctx["floor"] = rp.behaviour_floor(m["behaviour"], d["X"][test], d["y"][test])
    picks = rp.pick_clips(d, m["held"], args.n_clips, args.seed)
    # ---- off-target readouts (factored control): ridge probes on the probe folds 3-4, disjoint from knots and steered
    # clips; speed on constant-speed clips only (steer.off_target_probe's rule), start position (x, y) on all
    df = d["df"]
    vel = df["motion"].to_numpy() == "velocity"
    spd_probe = mf.ProbeReadout(d["X"][probe_rows & vel], df["speed_mps"].to_numpy(float)[probe_rows & vel], False)
    pos_xy = df[["start_x", "start_y"]].to_numpy(float)
    pos_probe = mf.ProbeReadout(d["X"][probe_rows], pos_xy[probe_rows], False)
    tv = test[vel[test]]
    off_quality = {"speed_probe_test_mae_mps": float(np.mean(np.abs(spd_probe.raw(d["X"][tv]).ravel()
                                                                    - df["speed_mps"].to_numpy(float)[tv]))),
                   "position_probe_test_mean_dist_m": float(np.mean(np.linalg.norm(
                       pos_probe.raw(d["X"][test]) - pos_xy[test], axis=1))),
                   "speed_sd_test_mps": float(df["speed_mps"].to_numpy(float)[tv].std()),
                   "start_pos_mean_dist_to_centroid_m": float(np.mean(np.linalg.norm(
                       pos_xy[test] - pos_xy[test].mean(0), axis=1)))}
    Qring = np.linalg.qr(gc.fit_circular_chart(d["X"][m["knot"]], d["y"][m["knot"]])["A"])[0]   # [D, 2]

    def offtarget(x, dend):
        xs = x + dend
        n2 = (dend ** 2).sum(1)
        return {"d_speed": np.abs(spd_probe.raw(xs).ravel() - spd_probe.raw(x).ravel()),
                "d_pos": np.linalg.norm(pos_probe.raw(xs) - pos_probe.raw(x), axis=1),
                "ring_frac": np.where(n2 > 0, ((dend @ Qring) ** 2).sum(1) / np.where(n2 > 0, n2, 1), np.nan)}

    knot = m["knot"]
    Zk, yk, kv = pca.project(d["X"][knot]), d["y"][knot], raw.values.astype(float)
    Wt = {float(t): w for t, w in zip(m["held"], cc.chord_weights(raw, m["held"]))}
    mode = args.and_mode
    # ---- aperture: COAST Stage 2 on knot clips only (all kept values as sources x the held-out targets)
    Wkept = cc.chord_weights(raw, kv)
    mean_ov, validity = {}, {}
    for a in cc.ALPHA_GRID:
        ov, bad = [], []
        for t, wt in Wt.items():
            Ct = cond_conceptor(Zk, yk, kv, wt, a, mode)
            for ws in Wkept[:: args.alpha_stride]:
                Cs = cond_conceptor(Zk, yk, kv, ws, a, mode)
                ov.append(cc.overlap(Ct, Cs))
                e = np.linalg.eigvalsh(cc.contrastive(Ct, Cs, "pinv"))
                bad.append((e.min() < -1e-6) or (e.max() > 1 + 1e-6))
        mean_ov[a], validity[a] = float(np.mean(ov)), float(np.mean(bad))
    alpha, in_band = cc.select_alpha(mean_ov)
    alphas_b = [alpha] + ([] if args.lite else [a for a in cc.ALPHA_GRID if a != alpha])

    K, rng = args.K, np.random.default_rng(args.seed + 1000)
    beta = np.linspace(0.0, 1.0, K)
    Vmu = pca.components @ pca.mean                                         # u = V x = z + V mean
    per = {}
    rand = {"random_a": [[] for _ in range(args.n_draws)], "random_b": [[] for _ in range(args.n_draws)]}
    rand_ref = {"coast_a": [], "coast_b": []}
    traces, pinv_bad, flanks = [], [], {}

    def add(arm, ev, ids, src, tgt, extra=None):
        a = per.setdefault(arm, {q: [] for q in (*KEEP, *OFF, "_wp_err", "_wp_radius", "_dend", "id", "src", "tgt",
                                                 "vel")})
        for q in (*KEEP, "_wp_err", "_wp_radius"):
            a[q].append(ev[q])
        for q, v in extra.items():
            a[q].append(v)
        a["_dend"].append(ev["_delta_end"])
        a["id"].append(ids)
        a["src"].append(src)
        a["tgt"].append(np.full(len(ids), tgt))

    for tgt, pick in picks.items():
        x, src = d["X"][pick].astype(float), d["y"][pick]
        ids = d["df"]["id"].to_numpy()[pick]
        Z, resid = pca.project(x), pca.complement(x)
        U = Z + Vmu
        flanks[str(tgt)] = {str(v): float(w) for v, w in zip(kv, Wt[tgt]) if w > 1e-9}
        Ct = {a: cond_conceptor(Zk, yk, kv, Wt[tgt], a, mode) for a in alphas_b}
        Wsrc = cc.chord_weights(raw, src)
        Cstack = {a: np.empty((len(pick), 64, 64)) for a in alphas_b}
        cache = {}
        for i, s in enumerate(src):
            if float(s) not in cache:
                cache[float(s)] = {}
                for a in alphas_b:
                    Cs = cond_conceptor(Zk, yk, kv, Wsrc[i], a, mode)
                    cache[float(s)][a] = cc.contrastive(Ct[a], Cs, mode)
                    if a == alpha:
                        e = np.linalg.eigvalsh(cc.contrastive(Ct[a], Cs, "pinv"))
                        pinv_bad.append((e.min() < -1e-6) or (e.max() > 1 + 1e-6))
            for a in alphas_b:
                Cstack[a][i] = cache[float(s)][a]
        C = Cstack[alpha]
        traces.append(np.trace(C, axis1=1, axis2=2))
        mu = mf.piecewise_linear_point(raw, np.full(len(pick), tgt))
        zarms = rp.subspace_arms(Z, src, tgt, m, K)
        paths = {"manifold": zarms["manifold"], "linear_raw": zarms["linear_raw"],
                 "coast_a": cc.gate_path(U, C, beta) - Vmu,
                 "coast_a_b0.3": cc.gate_path(U, C, 0.3 * beta) - Vmu,
                 **({} if args.lite else {"coast_a_b0.1": cc.gate_path(U, C, 0.1 * beta) - Vmu}),
                 "coast_a_centered": cc.gate_path(Z, C, beta),
                 "coast_b": cc.aimed_path(Z, C, mu, beta)}
        # coast_b rescaled per clip and waypoint to the raw chord's ||delta|| (is it the direction or the dose?)
        db, dl = paths["coast_b"] - Z[:, None], zarms["linear_raw"] - Z[:, None]
        nb, nl = np.linalg.norm(db, axis=-1, keepdims=True), np.linalg.norm(dl, axis=-1, keepdims=True)
        paths["coast_b_dose_matched"] = Z[:, None] + db * np.where(nb > 0, nl / np.where(nb > 0, nb, 1.0), 0.0)
        for a in (alphas_b[1:] if not args.lite else []):
            paths[f"coast_b_alpha{a:g}"] = cc.aimed_path(Z, Cstack[a], mu, beta)
        for arm, Zp in paths.items():
            ev = rp.evaluate(rp.compose(pca, Zp, resid), x, src, tgt, m, ctx)
            add(arm, ev, ids, src, tgt, {**offtarget(x, ev["_delta_end"]), "vel": vel[pick]})
        # ---- rank-matched random-projector null, first n_control_clips clips per target
        sub = slice(0, args.n_control_clips)
        r = np.maximum(1, np.rint(traces[-1][sub])).astype(int)
        for arm in rand_ref:
            rand_ref[arm].append({q: per[arm][q][-1][sub] for q in ("probe_err_to_target", "nearest_real_R",
                                                                     "probe_radius_min", "delta_norm")})
        for j in range(args.n_draws):
            P = np.stack([cc.random_projector(64, ri, rng) for ri in r])
            for arm, Zp in (("random_a", cc.gate_path(U[sub], P, beta) - Vmu),
                            ("random_b", cc.aimed_path(Z[sub], P, mu[sub], beta))):
                ev = rp.evaluate(rp.compose(pca, Zp, resid[sub]), x[sub], src[sub], tgt, m, ctx)
                ot = offtarget(x[sub], ev["_delta_end"])
                ot["d_speed"] = ot["d_speed"][vel[pick][sub]]
                rand[arm][j].append({**{q: ev[q] for q in ("probe_err_to_target", "nearest_real_R", "probe_radius_min",
                                                           "delta_norm", "_wp_err", "_wp_radius")}, **ot})

    cat = {a: {q: np.concatenate(v) for q, v in qs.items()} for a, qs in per.items()}
    chord = cat["linear_raw"]
    arms_out = {}
    for arm, c in cat.items():
        ids = c["id"]
        ratio = c["delta_norm"] / np.where(chord["delta_norm"] > 0, chord["delta_norm"], np.nan)
        arms_out[arm] = {
            "unsteered_err_deg": float(c["_wp_err"][:, 0].mean()),
            "endpoint_err_deg": stats(c["probe_err_to_target"], ids, args.seed),
            "gap_vs_raw_chord_deg": stats(c["probe_err_to_target"] - chord["probe_err_to_target"], ids, args.seed),
            "nearest_real_R": stats(c["nearest_real_R"], ids, args.seed),
            "probe_radius_min": float(np.mean(c["probe_radius_min"])),
            "ordering_spearman": float(np.nanmean(c["ordering_spearman"])),
            "probe_err_path_deg": float(np.mean(c["probe_err_path"])),
            "intermediate_mass": float(np.mean(c["intermediate_mass"])),
            "excess_to_nearest_real": float(np.mean(c["excess_to_nearest_real"])),
            "delta_norm": float(np.mean(c["delta_norm"])),
            "delta_norm_ratio_to_raw_chord": {"mean_of_ratios": float(np.nanmean(ratio)),
                                              "ratio_of_means": float(c["delta_norm"].mean() / chord["delta_norm"].mean())},
            "edit_participation_ratio": effective_rank(c["_dend"]),
            "off_target": {
                "abs_change_speed_mps": stats(c["d_speed"][c["vel"]], ids[c["vel"]], args.seed),
                "abs_change_start_position_m": stats(c["d_pos"], ids, args.seed),
                "ring_plane_energy_frac": stats(c["ring_frac"], ids, args.seed)},
            "wp_err_mean": c["_wp_err"].mean(0).tolist(), "wp_radius_mean": c["_wp_radius"].mean(0).tolist()}
    tr = np.concatenate(traces)
    for arm in [a for a in ("coast_a", "coast_a_b0.3", "coast_a_b0.1", "coast_a_centered", "coast_b",
                            "coast_b_dose_matched") if a in arms_out]:
        arms_out[arm]["alpha"] = alpha
        arms_out[arm]["trace_C_mean"] = float(tr.mean())
    for a in (alphas_b[1:] if not args.lite else []):
        arms_out[f"coast_b_alpha{a:g}"]["alpha"] = a
    null = {}
    for arm, draws in rand.items():
        ref = {q: np.concatenate([r_[q] for r_ in rand_ref["coast_a" if arm == "random_a" else "coast_b"]])
               for q in ("probe_err_to_target", "nearest_real_R", "probe_radius_min", "delta_norm")}
        dm = {q: np.array([np.concatenate([b[q] for b in dr]).mean() for dr in draws])
              for q in ("probe_err_to_target", "nearest_real_R", "probe_radius_min", "delta_norm")}
        null.setdefault(arm, {})["off_target"] = {q: {"draws_mean": float(v.mean()), "p05": float(np.percentile(v, 5)),
                                       "p95": float(np.percentile(v, 95))}
                                   for q, v in ((q, np.array([np.concatenate([b[q] for b in dr]).mean()
                                                              for dr in draws])) for q in OFF)}
        null[arm] = {**null.get(arm, {}), **{q: {"draws_mean": float(v.mean()), "p05": float(np.percentile(v, 5)),
                         "p95": float(np.percentile(v, 95)), "conceptor_same_clips": float(ref[q].mean()),
                         "frac_draws_conceptor_beats": float(np.mean(
                             v > ref[q].mean() if q == "probe_err_to_target" else v < ref[q].mean()))}
                     for q, v in dm.items()}}
        null[arm]["wp_err_mean"] = np.mean([np.concatenate([b["_wp_err"] for b in dr]).mean(0) for dr in draws],
                                           0).tolist()
        null[arm]["wp_radius_mean"] = np.mean([np.concatenate([b["_wp_radius"] for b in dr]).mean(0)
                                               for dr in draws], 0).tolist()
    null["rank"] = {"mean": float(np.mean(np.maximum(1, np.rint(tr)))), "rule": "round(trace C), at least 1"}
    null["note"] = (f"{args.n_draws} draws, first {args.n_control_clips} steered clips per target; "
                    "frac_draws_conceptor_beats = share of draws worse than the conceptor arm on the same clips")

    stored = PROJECT_ROOT / "results" / f"p2_steer_direction_direction_L{L}_contiguous_rawchord.json"
    if args.stored:
        stored = Path(args.stored)
    repro = None
    if stored.exists():
        s = json.loads(stored.read_text())["summary"]
        repro = {a: {"stored": s[a]["overall"]["probe_err_to_target"],
                     "recomputed": arms_out[a]["endpoint_err_deg"]["mean"]} for a in ("manifold", "linear_raw")}
        repro["source"] = str(stored.relative_to(PROJECT_ROOT) if stored.is_relative_to(PROJECT_ROOT) else stored)
    out = {"dataset": "direction", "layer": L, "seed": args.seed, "angle_source": m["curve"].coord_source,
           "provenance": provenance(None, seeds={"seed": args.seed, "random_projectors": args.seed + 1000}, layer=L,
                                    pool="meanpool", holdout="contiguous", spline="smooth", k=64, subspace="pca", K=K,
                                    n_clips=args.n_clips, and_mode=mode),
           "held_out_values": m["held"].tolist(), "holdout": m["design"],
           "target_condition_knots": flanks,
           "aperture": {"grid": list(cc.ALPHA_GRID), "mean_overlap": {str(k): v for k, v in mean_ov.items()},
                        "selected": alpha, "in_band": in_band, "band": list(cc.OVERLAP_BAND),
                        "rule": "COAST A.10.2 Stage 2: mean Eq. 11 overlap of C_target and C_source over all "
                                "(kept value, held-out target) pairs of knot clips; in band -> nearest band centre, "
                                "none in band -> closest (COAST's rule)",
                        "pinv_AND_invalid_frac_by_alpha": {str(k): v for k, v in validity.items()},
                        "pinv_AND_invalid_frac_steered_pairs": float(np.mean(pinv_bad)),
                        "and_mode_used": mode},
           "reproduction_check": repro, "arms": arms_out,
           "held_out_aim": args.aim,
           "per_clip": {arm: {"id": c["id"].tolist(), "tgt": c["tgt"].tolist(),
                              "probe_err_to_target": c["probe_err_to_target"].tolist()} for arm, c in cat.items()},
           "off_target_readouts": {
               **off_quality,
               "speed": "ridge (StandardScaler + RidgeCV) on probe folds 3-4, constant-speed clips only; |change| read "
                        "on the constant-speed steered clips, m/s",
               "start_position": "ridge on probe folds 3-4, (start_x, start_y); |change| = Euclidean distance, m",
               "ring_plane_energy_frac": "||P_ring delta||^2 / ||delta||^2, ring plane = span(A) of the circular chart "
                                         "X ~ mu + A [cos, sin] fit on the knot clips (geometry_checks."
                                         "fit_circular_chart); endpoint edit",
               "ci": "95% clip bootstrap (clips resampled with all their targets)"}, "random_projector_null": null,
           "config": {"K": K, "n_clips": args.n_clips, "n_draws": args.n_draws,
                      "n_control_clips": args.n_control_clips, "labels_angle": args.labels_angle,
                      "alpha_stride": args.alpha_stride, "lite": args.lite, "beta_schedule": "linspace(0, 1, K)",
                      "readouts": "run_part2.evaluate: ridge probe on probe folds 3-4, nearest-real R vs test clips "
                                  "at the target, probe (sin, cos) radius along the path, Eq. 9 ordering"}}
    path = Path(args.out) if args.out else PROJECT_ROOT / "results" / f"p2_conceptor_direction_L{L}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    if not args.no_fig:
        plot(out, PROJECT_ROOT / "figures" / f"fig_conceptor_direction_L{L}.png")
    return out


def plot(out, path):
    arms = [("manifold", "spline"), ("linear_raw", "raw chord"), ("coast_a", "conceptor a (COAST gate)"),
            ("coast_b", "conceptor b (target-aimed)")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    for key, lab in arms:
        a = out["arms"][key]
        f = np.linspace(0, 1, len(a["wp_err_mean"]))
        axes[0].plot(f, a["wp_err_mean"], label=lab)
        axes[1].plot(f, a["wp_radius_mean"], label=lab)
    for key, lab in (("random_b", "random projector, b update (16 clips/target)"),
                     ("random_a", "random projector, a update (16 clips/target)")):
        n = out["random_projector_null"][key]
        f = np.linspace(0, 1, len(n["wp_err_mean"]))
        axes[0].plot(f, n["wp_err_mean"], ":", color="0.4" if key == "random_b" else "0.7", label=lab)
        axes[1].plot(f, n["wp_radius_mean"], ":", color="0.4" if key == "random_b" else "0.7", label=lab)
    axes[0].set(xlabel="fraction of path (s or beta)", ylabel="probe error to target (deg)",
                title="Endpoint probe: error along the path")
    axes[1].set(xlabel="fraction of path (s or beta)", ylabel="probe readout radius",
                title="Probe (sin, cos) radius along the path")
    axes[0].legend(fontsize=7)
    # third panel: off-target endpoint effects (factored control)
    bars = arms + [("random_b", "random proj. (b)"), ("random_a", "random proj. (a)")]
    qs = (("abs_change_speed_mps", "d_speed", "|d speed| (m/s)"), ("abs_change_start_position_m", "d_pos",
          "|d start pos| (m)"), ("ring_plane_energy_frac", "ring_frac", "ring-plane share of ||delta||^2"))
    w = 0.8 / len(bars)
    for i, (key, lab) in enumerate(bars):
        if key in out["arms"]:
            o = out["arms"][key]["off_target"]
            mean = [o[q]["mean"] for q, _, _ in qs]
            err = np.array([[o[q]["mean"] - o[q]["ci95"][0], o[q]["ci95"][1] - o[q]["mean"]] for q, _, _ in qs]).T
        else:
            o = out["random_projector_null"][key]["off_target"]
            mean = [o[r]["draws_mean"] for _, r, _ in qs]
            err = np.array([[o[r]["draws_mean"] - o[r]["p05"], o[r]["p95"] - o[r]["draws_mean"]] for _, r, _ in qs]).T
        axes[2].bar(np.arange(len(qs)) + (i - len(bars) / 2 + 0.5) * w, mean, w, yerr=err, capsize=2, label=lab,
                    color=None if i < len(arms) else ("0.4" if key == "random_b" else "0.7"))
    axes[2].set_xticks(np.arange(len(qs)), [t for _, _, t in qs], fontsize=7)
    axes[2].set_yscale("log")
    axes[2].set(title="Off-target at endpoint (log; 95% CI, random 5-95%)")
    axes[2].legend(fontsize=6)
    ap = out["aperture"]
    fig.suptitle(f"Conceptor steering, direction, point {out['layer']}, held-out arc "
                 f"{out['held_out_values'][0]:g}-{out['held_out_values'][-1]:g} deg; alpha = {ap['selected']:g} "
                 f"(COAST overlap rule, mean overlap {max(ap['mean_overlap'].values()):.2f}, band 0.85-0.95 not met)"
                 if not ap["in_band"] else f"Conceptor steering, point {out['layer']}, alpha = {ap['selected']:g}",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def aggregate(L, arcs_dir):
    """Across-arc summary (seeds 1-16) added to the headline JSON under 'arcs'."""
    res = {}
    for s in range(1, 17):
        sub = f"L{L}_s{s}_labels" if L == 22 and s in LABELS22_SEEDS else f"L{L}_s{s}"
        p = Path(arcs_dir) / sub / f"p2_conceptor_direction_L{L}.json"
        if p.exists():
            res[s] = json.loads(p.read_text())
    if not res:
        return None
    arms = ("manifold", "linear_raw", "coast_a", "coast_a_b0.3", "coast_a_centered", "coast_b", "coast_b_dose_matched")
    summ = {"seeds": sorted(res), "n_arcs": len(res),
            "alpha_selected": {str(s): r["aperture"]["selected"] for s, r in res.items()},
            "in_band_any": any(r["aperture"]["in_band"] for r in res.values())}
    for a in arms:
        e = np.array([r["arms"][a]["endpoint_err_deg"]["mean"] for r in res.values()])
        g = np.array([r["arms"][a]["gap_vs_raw_chord_deg"]["mean"] for r in res.values()])
        gci = np.array([r["arms"][a]["gap_vs_raw_chord_deg"]["ci95"] for r in res.values()])
        R = np.array([r["arms"][a]["nearest_real_R"]["mean"] for r in res.values()])
        rad = np.array([r["arms"][a]["probe_radius_min"] for r in res.values()])
        nr = np.array([r["arms"][a]["delta_norm_ratio_to_raw_chord"]["ratio_of_means"] for r in res.values()])
        summ[a] = {"endpoint_err_mean": float(e.mean()), "endpoint_err_sd": float(e.std(ddof=1)),
                   "gap_vs_raw_chord_mean": float(g.mean()), "gap_vs_raw_chord_sd": float(g.std(ddof=1)),
                   "n_arcs_ci_above_zero": int((gci[:, 0] > 0).sum()), "n_arcs_ci_below_zero": int((gci[:, 1] < 0).sum()),
                   "nearest_real_R_mean": float(R.mean()), "probe_radius_min_mean": float(rad.mean()),
                   "norm_ratio_to_raw_chord_mean": float(nr.mean()),
                   **{f"{q}_mean": float(np.mean([r["arms"][a]["off_target"][q]["mean"] for r in res.values()]))
                      for q in ("abs_change_speed_mps", "abs_change_start_position_m", "ring_plane_energy_frac")}}
    for a in ("random_a", "random_b"):
        summ[a] = {"endpoint_err_draws_mean": float(np.mean([r["random_projector_null"][a]["probe_err_to_target"]
                                                             ["draws_mean"] for r in res.values()])),
                   "frac_draws_conceptor_beats_mean": float(np.mean(
                       [r["random_projector_null"][a]["probe_err_to_target"]["frac_draws_conceptor_beats"]
                        for r in res.values()]))}
    return summ


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--labels-angle", action="store_true")
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--n-clips", type=int, default=48)
    p.add_argument("--n-draws", type=int, default=20)
    p.add_argument("--n-control-clips", type=int, default=16)
    p.add_argument("--alpha-stride", type=int, default=1, help="every n-th kept value as a source for the overlap")
    p.add_argument("--aim", default="coord", choices=("coord", "arc"),
                   help="periodic held-out aim of the chord (and so of the conceptor conditions): arc = issue #213/#266 fix")
    p.add_argument("--and-mode", default="jaeger", choices=("jaeger", "pinv"))
    p.add_argument("--out", default=None)
    p.add_argument("--stored", default=None, help="run_part2 JSON whose spline / raw-chord numbers are checked")
    p.add_argument("--no-fig", action="store_true")
    p.add_argument("--lite", action="store_true", help="arcs: drop coast_a_b0.1 and the coast_b alpha sweep")
    p.add_argument("--replot", type=int, default=None, help="redraw the figure of point N from its JSON")
    p.add_argument("--aggregate", type=int, default=None, help="add the arcs summary to the headline JSON of point N")
    p.add_argument("--arcs-dir", default=str(PROJECT_ROOT / "results" / "arcs_conceptor"))
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse()
    if a.replot is not None:
        plot(json.loads((PROJECT_ROOT / "results" / f"p2_conceptor_direction_L{a.replot}.json").read_text()),
             PROJECT_ROOT / "figures" / f"fig_conceptor_direction_L{a.replot}.png")
    elif a.aggregate is not None:
        hp = PROJECT_ROOT / "results" / f"p2_conceptor_direction_L{a.aggregate}.json"
        h = json.loads(hp.read_text())
        h["arcs"] = aggregate(a.aggregate, a.arcs_dir)
        hp.write_text(json.dumps(h, indent=1))
    else:
        run(a)

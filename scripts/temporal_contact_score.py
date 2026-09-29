"""Scoring for scripts/run_temporal_contact.py (local CPU): results/p5_temporal_locality.json,
results/p5_contact_dynamics.json and their figures."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))  # run without installing wm
import json
from pathlib import Path

import numpy as np

from run_temporal_contact import (ART, CT_POINTS, CTX_GROUPS, FIG, FULL_GROUPS, POINTS, PROJECT_ROOT, RES,
                                  tubelet_phase, turn_fraction, wrap_signed)
from run_token_patching import summarize


def _probe_tools():
    from wm.probes import Standardizer, cv_select_alpha, fit_ridge, predict, score as pscore
    return Standardizer, cv_select_alpha, fit_ridge, predict, pscore


def folds_for(n, seed=0, k=5):
    return np.random.default_rng(seed).permutation(np.arange(n) % k)


def fit_probe(X, Y, kind, seed=0):
    """Standardise, CV-select alpha on 5 random folds, refit on all rows. Returns ((st, W, b), cv dict)."""
    Standardizer, cv_select_alpha, fit_ridge, _, pscore = _probe_tools()
    X = np.asarray(X, np.float64)
    st = Standardizer().fit(X)
    Z = st.transform(X)
    cv = cv_select_alpha(Z, Y, folds_for(len(X), seed), score_fn=lambda a, b: pscore(a, b, kind))
    W, b = fit_ridge(Z, Y, cv["alpha"])
    return (st, W, b), {"alpha": cv["alpha"], "cv_r2": cv["cv_mean"], "cv_r2_sd": cv["cv_sd"],
                        "cv_mae": cv["cv_mae_mean"], "n": int(len(X))}


def apply(pr, X):
    st, W, b = pr
    return st.transform(np.asarray(X, np.float64)) @ W + b


def ang(P):
    return np.degrees(np.arctan2(P[..., 0], P[..., 1])) % 360.0


def cerr(a, b):
    return np.abs(wrap_signed(np.asarray(a) - b))


def score(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from run_session2 import write
    from wm.data import load_table
    from wm.probes import load_activations, targets
    from wm.provenance import sha256_file

    root = Path(root)
    F = dict(np.load(root / "forward.npz"))
    finfo = json.loads((root / "forward_info.json").read_text())
    P = dict(np.load(ART / "plan.npz"))
    cfgs = json.loads((ART / "bounces.json").read_text())["configs"]
    df = load_table("direction")
    row = {int(i): k for k, i in enumerate(df["id"])}
    theta = df["theta_degrees"].to_numpy(float)
    Ydir = targets(df, "direction")[0]
    TPd = load_activations("direction", "timepool")
    sdf = load_table("speed")
    TPs = load_activations("speed", "timepool")
    Ysp = targets(sdf, "speed")[0]

    # ------------------------------------------------ 1a per-tubelet CV R^2
    loc = {"direction": {}, "speed": {}}
    for var, TP, Y, kind in (("direction", TPd, Ydir, "circular"), ("speed", TPs, Ysp, "scalar")):
        for p in POINTS:
            X = np.asarray(TP[:, p], np.float32)                      # [N, 8, D]
            per = [fit_probe(X[:, t], Y, kind)[1] for t in range(8)]
            mp = fit_probe(X.mean(1), Y, kind)[1]
            loc[var][str(p)] = {"per_tubelet_cv_r2": [c["cv_r2"] for c in per],
                                "per_tubelet_cv_r2_sd": [c["cv_r2_sd"] for c in per],
                                "per_tubelet_cv_mae": [c["cv_mae"] for c in per],
                                "time_mean_cv_r2": mp["cv_r2"], "alphas": [c["alpha"] for c in per],
                                "argmax_tubelet": int(np.argmax([c["cv_r2"] for c in per]))}
            print(var, p, np.round(loc[var][str(p)]["per_tubelet_cv_r2"], 3), flush=True)

    # ------------------------------------------------ 1b frame-group mean ablation at point 12
    prow = np.array([row[int(i)] for i in P["probe_ids"][:len(F["probe_enc_mp"])]])
    n_eval = len(F["abl/ctx/none"])
    erow = np.array([row[int(i)] for i in P["eval_ids"][:n_eval]])
    Zall = np.load(PROJECT_ROOT / "artifacts/session2/native/pred_pooled_all.npy").astype(np.float64)
    fc_probe, fc_info = fit_probe(Zall[prow].mean(1), Ydir[prow], "circular")
    eo_probe, eo_info = fit_probe(F["probe_enc_mp"], Ydir[prow], "circular")
    abl = {"readers": {"forecast_step_mean": fc_info, "encoder_output_post_LN_meanpool": eo_info},
           "parity_bf16_vs_native_fp32_forecast_deg": summarize(cerr(ang(apply(fc_probe, F["abl/ctx/none"].mean(1))),
                                                                     ang(apply(fc_probe, Zall[erow].mean(1))))),
           "forecast": {}, "encoder_output": {}}
    th_e = theta[erow]
    base_fc = cerr(ang(apply(fc_probe, F["abl/ctx/none"].mean(1))), th_e)
    base_eo = cerr(ang(apply(eo_probe, F["abl/full/none"])), th_e)
    for name in ["none"] + list(CTX_GROUPS):
        e = cerr(ang(apply(fc_probe, F[f"abl/ctx/{name}"].mean(1))), th_e)
        abl["forecast"][name] = {"err_deg": summarize(e), "acc15": float((e <= 15).mean()),
                                 "err_increase_vs_none_deg": summarize(e - base_fc),
                                 "tubelets_ablated": list(CTX_GROUPS.get(name, ()))}
    for name in ["none"] + list(FULL_GROUPS):
        e = cerr(ang(apply(eo_probe, F[f"abl/full/{name}"])), th_e)
        abl["encoder_output"][name] = {"err_deg": summarize(e), "acc15": float((e <= 15).mean()),
                                       "err_increase_vs_none_deg": summarize(e - base_eo),
                                       "tubelets_ablated": list(FULL_GROUPS.get(name, ()))}
    res1 = {"question": "at which frames of the 16-frame clip does the direction / speed code live?",
            "definitions": {
                "tubelet": "t = 0..7 covers frames 2t+1, 2t+2 (1-indexed); feature = spatial mean of its 256 tokens "
                           "(stored timepool of the full-clip fp32 encoding)",
                "cv": "5 random folds (seed 0), ridge alpha chosen by fold-mean R^2, standardiser fit on all rows; "
                      "direction R^2 = mean over (sin, cos); speed R^2 on speed_mps",
                "caveat": "full-clip encoder attention is bidirectional, so per-tubelet readability is not "
                          "per-frame information origin; 1b is the causal test",
                "ablation": "point 12 (input to block 13 in 1-indexed blocks = the residual after 12 blocks): tokens "
                            "of the named tubelets replaced by the per-position mean over the 480 probe-role clips "
                            "(8-frame context encoding for the forecast, 16-frame for the encoder output)",
                "err_deg": "circular |read direction - true theta| over 200 knot/test clips; increase is paired"},
            "per_tubelet_probe": loc, "ablation_point12": abl,
            "config": {"points": POINTS, "forward": finfo, "n_direction": int(len(df)), "n_speed": int(len(sdf))}}
    write(RES / "p5_temporal_locality.json", res1, seeds={"folds": 0, "eval_clips": 0, "bootstrap": 0},
          activation_sha256={"forward_npz": sha256_file(root / "forward.npz")}, script="scripts/run_temporal_contact.py")

    # ------------------------------------------------ 2 contact
    vel = np.where(df["motion"].to_numpy() == "velocity")[0]
    kb = np.array([c["k_b"] for c in cfgs])[:len(F["ct/bounce/timepool"])]
    th_in = np.array([c["theta_in"] for c in cfgs])[:len(kb)]
    th_out = np.array([c["theta_out"] for c in cfgs])[:len(kb)]
    phase = np.stack([tubelet_phase(k) for k in kb])                  # [n, 8]
    par = F["parity_timepool"]
    prow_par = np.array([row[int(i)] for i in F["parity_ids"]])
    stored = np.asarray(TPd[prow_par][:, list(CT_POINTS)], np.float32)
    parity = {str(p): float(np.abs(par[:, j] - stored[:, j]).max() / np.abs(stored[:, j]).max())
              for j, p in enumerate(CT_POINTS)}
    ct = {"n_bounce": int(len(kb)), "parity_fp32_timepool_rel_maxabs_vs_stored": parity, "per_point": {}}
    tf_curves, tfs_all = {}, {}
    for j, p in enumerate(CT_POINTS):
        X = np.asarray(TPd[vel][:, p], np.float32)
        R = {"probe_cv_r2_by_tubelet": [], "bounce": {}, "straight": {}}
        tfs = np.full((len(kb), 8), np.nan)
        e_in, e_out, e_str = np.full((len(kb), 8), np.nan), np.full((len(kb), 8), np.nan), np.full((len(kb), 8), np.nan)
        for t in range(8):
            pr, info = fit_probe(X[:, t], Ydir[vel], "circular")
            R["probe_cv_r2_by_tubelet"].append(info["cv_r2"])
            phb = ang(apply(pr, F["ct/bounce/timepool"][:, j, t]))
            phs = ang(apply(pr, F["ct/straight/timepool"][:, j, t]))
            tfs[:, t] = turn_fraction(phb, th_in, th_out)
            e_in[:, t], e_out[:, t], e_str[:, t] = cerr(phb, th_in), cerr(phb, th_out), cerr(phs, th_in)
        for ph, nm in ((-1, "pre_contact"), (0, "straddle"), (1, "post_contact")):
            m = phase == ph
            R["bounce"][nm] = {"turn_fraction": summarize(tfs[m]), "err_to_theta_in": summarize(e_in[m]),
                               "err_to_theta_out": summarize(e_out[m]),
                               "frac_closer_to_out": float((e_out[m] < e_in[m]).mean()), "n_cells": int(m.sum())}
            R["straight"][nm] = {"err_to_theta_in": summarize(e_str[m])}
        # by offset from contact tubelet (t - floor(k_b / 2))
        off = np.arange(8)[None] - (kb // 2)[:, None]
        R["bounce"]["turn_fraction_by_offset"] = {str(o): summarize(tfs[off == o]) for o in range(-6, 7)
                                                  if (off == o).sum() >= 5}
        # post-contact tubelets split by time since contact (does the code catch up?)
        tf_curves[p] = R["bounce"]["turn_fraction_by_offset"]
        tfs_all[p] = tfs
        ct["per_point"][str(p)] = R
    # (b) predictor forecast, contact inside the forecast window
    fut = kb >= 8
    fc_step = []
    for s in range(4):
        pr, info = fit_probe(Zall[prow][:, s], Ydir[prow], "circular")
        fc_step.append((pr, info))
    B = {"n_clips_contact_in_forecast_window": int(fut.sum()), "step_probe_cv_r2": [i["cv_r2"] for _, i in fc_step],
         "steps": {}}
    for s in range(4):
        t = 4 + s
        post = fut & (phase[:, t] == 1)
        phf = ang(apply(fc_step[s][0], F["ct/bounce/forecast"][:, s]))
        phfs = ang(apply(fc_step[s][0], F["ct/straight/forecast"][:, s]))
        j25 = CT_POINTS.index(25)
        # encoder's real-future tubelet read with the point-25 per-tubelet probe (from (a)) as a reference
        B["steps"][f"step{s}_tubelet{t}"] = {
            "n_post_contact": int(post.sum()),
            "forecast_turn_fraction_post": summarize(turn_fraction(phf, th_in, th_out)[post]),
            "forecast_err_to_theta_in_post": summarize(cerr(phf, th_in)[post]),
            "forecast_err_to_theta_out_post": summarize(cerr(phf, th_out)[post]),
            "forecast_frac_closer_to_out_post": float((cerr(phf, th_out) < cerr(phf, th_in))[post].mean()) if post.any() else None,
            "forecast_err_to_theta_in_pre_or_straddle": summarize(cerr(phf, th_in)[fut & (phase[:, t] < 1)]),
            "straight_twin_forecast_err_to_theta_in": summarize(cerr(phfs, th_in)),
            "real_future_encoder_turn_fraction_post_point25": summarize(tfs_all[25][post, t])}
    ct["predictor_forecast"] = B
    res2 = {"question": "does the V-JEPA 2 direction code change at a wall bounce, and does the predictor forecast "
                        "the reflected direction?",
            "data": {"supplied_sets_have_bounces": False,
                     "rendered": "96 wall-bounce clips + 96 straight twins (wm.render_twin params, MPEG-4 codec), "
                                 "grey wall bar 6 px, elastic reflection at integer contact frame k_b in 3..12 "
                                 "(0-indexed), speeds 2/3/4 m/s, incidence 20-65 deg from the normal",
                     "configs_file": "artifacts/p5_temporal_contact/bounces.json",
                     "k_b_counts": {str(k): int((kb == k).sum()) for k in np.unique(kb)}},
            "definitions": {
                "turn_fraction": "signed angular move of the read angle from theta_in along the shorter arc toward "
                                 "theta_out, over that arc (0 = incoming, 1 = reflected)",
                "phase": "tubelet t (frames 2t, 2t+1, 0-indexed) pre_contact if 2t+1 < k_b, post_contact if 2t >= k_b",
                "probes_a": "per-tubelet ridge direction probes on the 750 constant-velocity supplied clips (stored "
                            "fp32 timepool), applied to the same tubelet of the rendered clips",
                "probes_b": "per-future-step ridge direction probes on the native fp32 forecast cache of the 480 "
                            "probe-role clips; bounce clips with k_b >= 8 (context frames 0-7 all pre-contact)"},
            "results": ct, "config": {"points": CT_POINTS, "forward": finfo}}
    write(RES / "p5_contact_dynamics.json", res2, seeds={"bounces": 0, "folds": 0, "bootstrap": 0},
          activation_sha256={"forward_npz": sha256_file(root / "forward.npz"),
                             "bounces_json": sha256_file(ART / "bounces.json")}, script="scripts/run_temporal_contact.py")

    # ------------------------------------------------ figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for p in POINTS:
        ax[0].plot(range(1, 9), loc["direction"][str(p)]["per_tubelet_cv_r2"], "o-", label=f"point {p}")
        ax[1].plot(range(1, 9), loc["speed"][str(p)]["per_tubelet_cv_r2"], "o-", label=f"point {p}")
    for a, t in zip(ax[:2], ("direction", "speed")):
        a.set_xticks(range(1, 9), [f"{2 * t - 1}-{2 * t}" for t in range(1, 9)], fontsize=8)
        a.set_xlabel("frames (tubelet)"); a.set_ylabel("5-fold CV R^2"); a.set_title(f"{t}: per-tubelet probe")
        a.legend(fontsize=8)
    names = ["none"] + list(CTX_GROUPS)
    m = [abl["forecast"][n]["err_deg"]["mean"] for n in names]
    ci = np.array([abl["forecast"][n]["err_deg"]["ci95"] for n in names])
    names2 = ["none", "first4_f1-4", "middle8_f5-12", "last4_f13-16"]
    m2 = [abl["encoder_output"][n]["err_deg"]["mean"] for n in names2]
    ci2 = np.array([abl["encoder_output"][n]["err_deg"]["ci95"] for n in names2])
    x = np.arange(len(names))
    ax[2].bar(x, m, yerr=[np.array(m) - ci[:, 0], ci[:, 1] - np.array(m)], color="C0", label="predictor forecast")
    x2 = np.arange(len(names2)) + len(names) + 0.8
    ax[2].bar(x2, m2, yerr=[np.array(m2) - ci2[:, 0], ci2[:, 1] - np.array(m2)], color="C1", label="encoder output")
    ax[2].set_xticks(list(x) + list(x2), names + names2, rotation=45, ha="right", fontsize=7)
    ax[2].set_ylabel("direction error (deg)"); ax[2].set_title("mean-ablate frames at point 12"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig_temporal_locality.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for p in CT_POINTS:
        c = tf_curves[p]
        ks = sorted(c, key=int)
        ax[0].errorbar([int(k) for k in ks], [c[k]["mean"] for k in ks],
                       yerr=np.abs(np.array([c[k]["ci95"] for k in ks]).T - [c[k]["mean"] for k in ks]),
                       marker="o", capsize=2, label=f"point {p}")
    ax[0].axhline(0, c="k", lw=.5); ax[0].axhline(1, c="k", lw=.5, ls="--"); ax[0].axvline(0, c="grey", lw=.5)
    ax[0].set_xlabel("tubelet - contact tubelet"); ax[0].set_ylabel("turn fraction (0 = in, 1 = reflected)")
    ax[0].set_title("encoder per-tubelet direction readout at a wall bounce"); ax[0].legend(fontsize=8)
    st = [B["steps"][k] for k in sorted(B["steps"])]
    mm = [s["forecast_turn_fraction_post"]["mean"] if s["forecast_turn_fraction_post"]["n"] else np.nan for s in st]
    ax[1].bar(range(4), mm, color="C2", label="predictor forecast (post-contact steps)")
    ax[1].plot(range(4), [s["real_future_encoder_turn_fraction_post_point25"]["mean"] for s in st], "o--", c="C1",
               label="encoder real future, same tubelets (point 25)")
    ax[1].axhline(0, c="k", lw=.5); ax[1].axhline(1, c="k", lw=.5, ls="--")
    ax[1].set_xticks(range(4), [f"step {s}\n(frames {9 + 2 * s}-{10 + 2 * s})" for s in range(4)], fontsize=8)
    ax[1].set_ylabel("turn fraction"); ax[1].set_title(f"forecast from frames 1-8, contact in future (n={int(fut.sum())})")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig_contact_dynamics.png", dpi=130); plt.close(fig)
    print("figures written")

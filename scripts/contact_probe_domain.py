"""Probe-domain check for the contact forecasts (CPU only, cached forwards; no new GPU work).

The per-step forecast direction reader of results/p5_contact_dynamics.json was fit only on forecasts of supplied
constant-velocity / constant-acceleration clips, so "the predictor does not turn" and "the reader cannot read a
turned forecast" were confounded. Three checks on the cached forecasts of both rendered sets:
  OUT = artifacts/p5_temporal_contact/out/forward.npz   (stored set: contact k_b 3..12; forecast analysis k_b >= 8)
  IN  = artifacts/p5_contact_in_context/out/forward.npz (in-window set: contact k_b 2..6)

(a) READER REFITS (step-pooled forecast -> [sin, cos], ridge, standardiser + alpha re-chosen per fit by 5-fold CV):
    native      the stored reader (480 native probe-role forecasts only)
    +straight   native + rendered forecasts of straight clips only (OUT straight twins, IN straight_in and
                straight_out): render domain and turned headings, but no bounce forecast in the fit
    +all_xfit   native + all rendered forecasts incl. bounces, labelled with the true heading of that future step
                (OUT: theta_in pre-contact, theta_out post-contact, straddle steps dropped; IN bounce: theta_out),
                5-fold cross-fit grouped by configuration; each clip scored by the fold whose fit excluded it
    transfer    OUT scored by native + all IN rows (IN bounces included; no OUT bounce label ever used), and IN
                scored by native + all OUT rows
    Turn fraction as in the stored set (0 = incoming, 1 = reflected); OUT on post-contact steps of k_b >= 8 clips.
(b) PROBE-FREE: the forecast step s compared with the encoder's own post-LN (point 25) encoding of the real future
    tubelet 4+s of the bounce clip (B) and of its straight twin (S): lambda = <F - S, B - S> / |B - S|^2 (0 = the
    straight-continuation future, 1 = the reflected future) and the fraction of steps nearer B than S (L2). B and S
    also differ by the wall bar (only bounce clips have one); on IN, straight_out's future (SO) has the bounce's
    exact positions and heading but no wall, so B - SO is the wall's own effect. "wall-projected" removes the
    per-step mean IN wall direction mean(B - SO) from all vectors before lambda.
(c) PRE-CONTACT PULL: turn fraction of pre-contact tubelets at point 22 (stored-set per-tubelet readers) against the
    number of post-contact frames the encoding can attend to (full 16-frame clip: 16 - k_b; context-only: 8 - k_b).

  python scripts/contact_probe_domain.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_temporal_contact import CT_POINTS, tubelet_phase, turn_fraction  # noqa: E402
from wm.data import PROJECT_ROOT  # noqa: E402

OUT_DIR = PROJECT_ROOT / "artifacts/p5_temporal_contact"
IN_DIR = PROJECT_ROOT / "artifacts/p5_contact_in_context"
RES, FIG = PROJECT_ROOT / "results", PROJECT_ROOT / "figures"
J25, J22 = CT_POINTS.index(25), CT_POINTS.index(22)


def lam(F, B, S):
    """Projection coefficient of F on the segment S -> B, rowwise ([n, D])."""
    d = B - S
    return np.einsum("nd,nd->n", F - S, d) / np.einsum("nd,nd->n", d, d)


def sc(th_deg):
    t = np.radians(np.asarray(th_deg, float))
    return np.stack([np.sin(t), np.cos(t)], -1)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from run_session2 import write
    from run_token_patching import summarize
    from temporal_contact_score import ang, apply, cerr, fit_probe
    from wm.data import load_table
    from wm.p2_data import load_inputs
    from wm.probes import load_activations, targets
    from wm.provenance import sha256_file

    FO = dict(np.load(OUT_DIR / "out/forward.npz"))
    FI = dict(np.load(IN_DIR / "out/forward.npz"))
    co = json.loads((OUT_DIR / "bounces.json").read_text())["configs"]
    ci = json.loads((IN_DIR / "configs.json").read_text())["configs"]
    kbo, kbi = np.array([c["k_b"] for c in co]), np.array([c["k_b"] for c in ci])
    tio, too = np.array([c["theta_in"] for c in co]), np.array([c["theta_out"] for c in co])
    tii, toi = np.array([c["theta_in"] for c in ci]), np.array([c["theta_out"] for c in ci])
    no, ni = len(co), len(ci)
    pho = np.stack([tubelet_phase(k) for k in kbo])
    phi_ = np.stack([tubelet_phase(k) for k in kbi])
    fut = kbo >= 8

    df = load_table("direction")
    Ydir = targets(df, "direction")[0]
    role = load_inputs("direction", 0)["role"]
    prow = np.where(role == "probe")[0]
    Zall = np.load(PROJECT_ROOT / "artifacts/session2/native/pred_pooled_all.npy").astype(np.float64)

    # ---------------------------------------------------------------- rendered rows per step
    def rows(step, which, groups_keep):
        """Rendered (X, Y) for one step. which: set of {'out_straight','out_bounce','in_straight','in_bounce'}."""
        X, Y = [], []
        t = 4 + step
        go, gi = groups_keep[:no], groups_keep[no:]
        if "out_straight" in which:
            X.append(FO["ct/straight/forecast"][go, step]); Y.append(sc(tio[go]))
        if "out_bounce" in which:
            m = go & (pho[:, t] != 0)
            X.append(FO["ct/bounce/forecast"][m, step]); Y.append(sc(np.where(pho[m, t] == 1, too[m], tio[m])))
        if "in_straight" in which:
            X += [FI["straight_in/forecast"][gi, step], FI["straight_out/forecast"][gi, step]]
            Y += [sc(tii[gi]), sc(toi[gi])]
        if "in_bounce" in which:
            X.append(FI["bounce/forecast"][gi, step]); Y.append(sc(toi[gi]))
        return np.concatenate(X), np.concatenate(Y)

    def reader(step, which, keep):
        X = Zall[prow][:, step]
        Y = Ydir[prow]
        if which:
            Xr, Yr = rows(step, which, keep)
            X, Y = np.concatenate([X, Xr]), np.concatenate([Y, Yr])
        return fit_probe(X, Y, "circular")

    allk = np.ones(no + ni, bool)
    folds = np.random.default_rng(0).permutation(np.arange(no + ni) % 5)
    ALL = {"out_straight", "out_bounce", "in_straight", "in_bounce"}
    variants = {"native": None, "+straight": {"out_straight", "in_straight"}, "+all_xfit": ALL,
                "transfer": "transfer"}
    # read angles [variant][set/kind] -> [n, 4]
    reads = {v: {k: np.full((no if k.startswith("out") else ni, 4), np.nan)
                 for k in ("out_bounce", "out_straight", "in_bounce", "in_straight_in", "in_straight_out")}
             for v in variants}
    fit_info = {v: [] for v in variants}

    def score_into(v, s, pr, mo, mi):
        R = reads[v]
        R["out_bounce"][mo, s] = ang(apply(pr, FO["ct/bounce/forecast"][mo, s]))
        R["out_straight"][mo, s] = ang(apply(pr, FO["ct/straight/forecast"][mo, s]))
        R["in_bounce"][mi, s] = ang(apply(pr, FI["bounce/forecast"][mi, s]))
        R["in_straight_in"][mi, s] = ang(apply(pr, FI["straight_in/forecast"][mi, s]))
        R["in_straight_out"][mi, s] = ang(apply(pr, FI["straight_out/forecast"][mi, s]))

    for s in range(4):
        for v, which in variants.items():
            if which is None:
                pr, info = reader(s, None, allk)
                score_into(v, s, pr, np.ones(no, bool), np.ones(ni, bool))
                fit_info[v].append(info)
            elif which == "transfer":
                keep_in = np.r_[np.zeros(no, bool), np.ones(ni, bool)]
                pr, info = reader(s, {"in_straight", "in_bounce"}, keep_in)
                score_into(v, s, pr, np.ones(no, bool), np.zeros(ni, bool))
                keep_out = np.r_[np.ones(no, bool), np.zeros(ni, bool)]
                pr2, info2 = reader(s, {"out_straight", "out_bounce"}, keep_out)
                score_into(v, s, pr2, np.zeros(no, bool), np.ones(ni, bool))
                fit_info[v].append({"out_scored_by_in_rows": info, "in_scored_by_out_rows": info2})
            else:
                for f in range(5):
                    keep = folds != f
                    pr, info = reader(s, which, keep)
                    score_into(v, s, pr, ~keep[:no], ~keep[no:])
                    fit_info[v].append({"step": s, "fold": f, **info})
            print("step", s, v, flush=True)

    A = {}
    for v in variants:
        R = reads[v]
        blk = {"out_of_window_k_b>=8": {}, "in_window": {}}
        for s in range(4):
            post = fut & (pho[:, 4 + s] == 1)
            tfb = turn_fraction(R["out_bounce"][:, s], tio, too)[post]
            tfs = turn_fraction(R["out_straight"][:, s], tio, too)[post]
            blk["out_of_window_k_b>=8"][f"step{s}"] = {
                "n_post_contact": int(post.sum()), "bounce_turn_fraction": summarize(tfb),
                "straight_twin_turn_fraction": summarize(tfs),
                "straight_twin_err_to_theta_in_deg": summarize(cerr(R["out_straight"][:, s], tio)[post])}
            blk["in_window"][f"step{s}"] = {
                "bounce_turn_fraction": summarize(turn_fraction(R["in_bounce"][:, s], tii, toi)),
                "straight_in_turn_fraction": summarize(turn_fraction(R["in_straight_in"][:, s], tii, toi)),
                "straight_out_turn_fraction": summarize(turn_fraction(R["in_straight_out"][:, s], tii, toi))}
        # pooled over post-contact steps (clip-level mean, then bootstrap over clips)
        tfo = np.where(fut[:, None] & (pho[:, 4:] == 1), turn_fraction(R["out_bounce"], tio[:, None], too[:, None]),
                       np.nan)
        keep = np.isfinite(tfo).any(1)
        blk["out_of_window_k_b>=8"]["clip_mean_post_steps"] = summarize(np.nanmean(tfo[keep], 1))
        blk["in_window"]["clip_mean_steps"] = summarize(turn_fraction(R["in_bounce"], tii[:, None], toi[:, None]).mean(1))
        A[v] = blk
    fit_summary = {}
    for v in variants:
        if v == "+all_xfit" or v == "+straight":
            fit_summary[v] = {"cv_r2_mean_over_folds_by_step": [float(np.mean([i["cv_r2"] for i in fit_info[v]
                                                                               if i["step"] == s])) for s in range(4)],
                              "n_rows_fold0_by_step": [next(i["n"] for i in fit_info[v] if i["step"] == s and
                                                            i["fold"] == 0) for s in range(4)]}
        elif v == "native":
            fit_summary[v] = {"cv_r2_by_step": [i["cv_r2"] for i in fit_info[v]], "n": fit_info[v][0]["n"]}
        else:
            fit_summary[v] = {"cv_r2_by_step": [[i["out_scored_by_in_rows"]["cv_r2"], i["in_scored_by_out_rows"]["cv_r2"]]
                                                for i in fit_info[v]]}

    # ---------------------------------------------------------------- (b) probe-free nearest future
    B_in, SI_in, SO_in = (FI[f"{k}/tp16"][:, J25, 4:] for k in ("bounce", "straight_in", "straight_out"))
    wall_dir = (B_in - SO_in).mean(0)                                           # [4, D] mean wall effect per step
    wall_u = wall_dir / np.linalg.norm(wall_dir, axis=1, keepdims=True)

    def proj_out(X, s):
        return X - (X @ wall_u[s])[:, None] * wall_u[s][None]

    PF = {"definition": "lambda = <F - S, B - S> / |B - S|^2 on step-pooled forecast F vs point-25 (post-LN) "
                        "timepool of the real future tubelet 4+s; B = bounce clip, S = straight twin",
          "out_of_window_k_b>=8": {}, "in_window": {}}
    B_o, S_o = FO["ct/bounce/timepool"][:, J25, 4:], FO["ct/straight/timepool"][:, J25, 4:]
    lam_o_clip, lam_i_clip = [], []
    for s in range(4):
        post = fut & (pho[:, 4 + s] == 1)
        Fb, Fs = FO["ct/bounce/forecast"][post, s], FO["ct/straight/forecast"][post, s]
        B, S = B_o[post, s], S_o[post, s]
        lb, ls_ = lam(Fb, B, S), lam(Fs, B, S)
        lbw = lam(proj_out(Fb, s), proj_out(B, s), proj_out(S, s))
        PF["out_of_window_k_b>=8"][f"step{s}"] = {
            "n": int(post.sum()), "lambda_bounce_forecast": summarize(lb),
            "lambda_straight_twin_forecast": summarize(ls_),
            "lambda_bounce_forecast_wall_projected": summarize(lbw),
            "frac_bounce_forecast_nearer_B_than_S": float((np.linalg.norm(Fb - B, axis=1) <
                                                           np.linalg.norm(Fb - S, axis=1)).mean()),
            "cos_forecast_vs_B_mean": float(np.mean(np.einsum("nd,nd->n", Fb, B) /
                                                    (np.linalg.norm(Fb, axis=1) * np.linalg.norm(B, axis=1))))}
        Fi = FI["bounce/forecast"][:, s]
        B, S, SO = B_in[:, s], SI_in[:, s], SO_in[:, s]
        li, lwi = lam(Fi, B, S), lam(proj_out(Fi, s), proj_out(B, s), proj_out(S, s))
        # wall-free reference: the same forecast on the straight_in -> straight_out segment (same headings, no wall)
        lso = lam(Fi, SO, S)
        d = np.stack([np.linalg.norm(Fi - X, axis=1) for X in (B, S, SO)], 1)
        PF["in_window"][f"step{s}"] = {
            "lambda_bounce_forecast": summarize(li), "lambda_bounce_forecast_wall_projected": summarize(lwi),
            "lambda_on_straight_in_to_straight_out_segment": summarize(lso),
            "lambda_straight_in_forecast": summarize(lam(FI["straight_in/forecast"][:, s], B, S)),
            "lambda_straight_out_forecast_on_SI_SO": summarize(lam(FI["straight_out/forecast"][:, s], SO, S)),
            "nearest_of_B_S_SO_fraction": {k: float((d.argmin(1) == j).mean()) for j, k in
                                           enumerate(("bounce_future", "straight_in_future", "straight_out_future"))},
            "frac_nearer_B_than_S": float((d[:, 0] < d[:, 1]).mean()),
            "wall_share_of_B_minus_S_norm": summarize(np.linalg.norm(B - SO, axis=1) / np.linalg.norm(B - S, axis=1))}
        lam_i_clip.append(li)
    lo = np.full((no, 4), np.nan)
    for s in range(4):
        post = fut & (pho[:, 4 + s] == 1)
        lo[post, s] = lam(FO["ct/bounce/forecast"][post, s], B_o[post, s], S_o[post, s])
    k_ = np.isfinite(lo).any(1)
    PF["out_of_window_k_b>=8"]["clip_mean_lambda_post_steps"] = summarize(np.nanmean(lo[k_], 1))
    PF["in_window"]["clip_mean_lambda"] = summarize(np.mean(lam_i_clip, 0))

    # ---------------------------------------------------------------- (c) pre-contact pull vs visible future frames
    TPd = load_activations("direction", "timepool")
    vel = np.where(df["motion"].to_numpy() == "velocity")[0]
    pre_cells = []                                  # (source, visible_post_frames, tf)
    for t in range(8):
        pr, _ = fit_probe(np.asarray(TPd[vel][:, 22, t], np.float32), Ydir[vel], "circular")
        for src, F_, kb, ti, to, ph, key in (("stored_full16", FO, kbo, tio, too, pho, "ct/bounce/timepool"),
                                             ("inwindow_full16", FI, kbi, tii, toi, phi_, "bounce/tp16")):
            m = ph[:, t] == -1
            if m.any():
                tf = turn_fraction(ang(apply(pr, F_[key][m, J22, t])), ti[m], to[m])
                pre_cells += [(src, 16 - k, v) for k, v in zip(kb[m], tf)]
    # context-only reader (fit on the context-only encodings of the velocity clips, as run_contact_in_context)
    for t in range(4):
        pr, _ = fit_probe(FI["vel/ctx8"][:, J22, t], Ydir[vel], "circular")
        m = phi_[:, t] == -1
        if m.any():
            tf = turn_fraction(ang(apply(pr, FI["bounce/ctx8"][m, J22, t])), tii[m], toi[m])
            pre_cells += [("inwindow_ctx8", 8 - k, v) for k, v in zip(kbi[m], tf)]
    src = np.array([c[0] for c in pre_cells]); vis = np.array([c[1] for c in pre_cells]); tfv = np.array([c[2] for c in pre_cells])
    PC = {"definition": "pre-contact tubelets (both frames before k_b), point 22, turn fraction vs number of "
                        "post-contact frames inside the encoded window (16 - k_b full clip, 8 - k_b context-only); "
                        "cells are (clip, tubelet), CI bootstraps cells", "by_source_and_visible": {}}
    for sname in ("stored_full16", "inwindow_full16", "inwindow_ctx8"):
        m = src == sname
        PC["by_source_and_visible"][sname] = {str(v): summarize(tfv[m & (vis == v)]) for v in np.unique(vis[m])
                                              if (m & (vis == v)).sum() >= 5}
        PC["by_source_and_visible"][sname]["all"] = summarize(tfv[m])
    slope = np.polyfit(vis, tfv, 1)
    rng = np.random.default_rng(0)
    bs = np.array([np.polyfit(vis[ix], tfv[ix], 1) for ix in rng.integers(0, len(vis), (2000, len(vis)))])
    PC["pooled_linear_fit_tf_vs_visible_post_frames"] = {
        "slope_per_frame": float(slope[0]), "slope_ci95": [float(np.percentile(bs[:, 0], 2.5)), float(np.percentile(bs[:, 0], 97.5))],
        "intercept_at_0_visible": float(slope[1]),
        "intercept_ci95": [float(np.percentile(bs[:, 1], 2.5)), float(np.percentile(bs[:, 1], 97.5))], "n_cells": int(len(vis))}

    res = {"question": "is 'the predictor does not forecast a bounce' a probe-domain artifact? (a) forecast reader "
                       "refit with rendered bounce forecasts (held out), (b) probe-free nearest real future, (c) is "
                       "the pre-contact encoder 'turn' future-frame leakage",
           "definitions": {"variants": {"native": "stored reader: 480 native probe-role forecasts",
                                        "+straight": "native + rendered straight forecasts (OUT straight twins, IN "
                                                     "straight_in + straight_out), 5-fold grouped cross-fit",
                                        "+all_xfit": "native + all rendered forecasts incl. bounces labelled with the "
                                                     "true heading of the step (OUT straddle steps dropped), 5-fold "
                                                     "cross-fit grouped by configuration (seed 0)",
                                        "transfer": "OUT scored by native + all IN rows; IN scored by native + all "
                                                    "OUT rows"},
                           "turn_fraction": "0 = incoming heading, 1 = reflected (run_temporal_contact.turn_fraction)",
                           "OUT": "p5_contact_dynamics set, forecast analysis on post-contact steps of k_b >= 8 clips",
                           "IN": "p5_contact_in_context set (k_b 2..6), all four steps post-contact",
                           "ci95": "bootstrap over clips (cells for (c)), 2000 resamples, seed 0"},
           "results": {"a_reader_refits": A, "a_fit_summary": fit_summary, "b_probe_free": PF,
                       "c_pre_contact_pull": PC},
           "config": {"n_out": no, "n_out_k_b>=8": int(fut.sum()), "n_in": ni, "gpu_seconds": 0}}
    write(RES / "p5_contact_probe_domain.json", res, seeds={"folds": 0, "bootstrap": 0},
          activation_sha256={"out_forward_npz": sha256_file(OUT_DIR / "out/forward.npz"),
                             "in_forward_npz": sha256_file(IN_DIR / "out/forward.npz"),
                             "out_bounces_json": sha256_file(OUT_DIR / "bounces.json"),
                             "in_configs_json": sha256_file(IN_DIR / "configs.json")},
          script="scripts/contact_probe_domain.py", script_sha256=sha256_file(Path(__file__)))

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = {"native": "C0", "+straight": "C1", "+all_xfit": "C3", "transfer": "C2"}
    for v in variants:
        for setn, mk, ls in (("out_of_window_k_b>=8", "o", "-"), ("in_window", "s", "--")):
            key = "bounce_turn_fraction"
            m = [A[v][setn][f"step{s}"][key]["mean"] for s in range(4)]
            ax[0].plot(range(4), m, marker=mk, ls=ls, c=cols[v], label=f"{v} ({'OUT' if setn[0] == 'o' else 'IN'})")
    ax[0].set_title("forecast turn fraction by reader (bounce clips)"); ax[0].legend(fontsize=6, ncol=2)
    ax[0].set_xticks(range(4), [f"step {s}" for s in range(4)])
    for setn, mk in (("out_of_window_k_b>=8", "o"), ("in_window", "s")):
        m = [PF[setn][f"step{s}"]["lambda_bounce_forecast"]["mean"] for s in range(4)]
        c = np.array([PF[setn][f"step{s}"]["lambda_bounce_forecast"]["ci95"] for s in range(4)])
        ax[1].errorbar(range(4), m, yerr=[np.array(m) - c[:, 0], c[:, 1] - np.array(m)], marker=mk, capsize=3,
                       label=f"{'OUT' if setn[0] == 'o' else 'IN'} raw")
        m = [PF[setn][f"step{s}"]["lambda_bounce_forecast_wall_projected"]["mean"] for s in range(4)]
        ax[1].plot(range(4), m, marker=mk, ls=":", label=f"{'OUT' if setn[0] == 'o' else 'IN'} wall-projected")
    ax[1].set_title("probe-free: forecast on straight(0) -> reflected(1) future"); ax[1].legend(fontsize=7)
    ax[1].set_xticks(range(4), [f"step {s}" for s in range(4)])
    for sname, c in (("stored_full16", "C0"), ("inwindow_full16", "C1"), ("inwindow_ctx8", "C2")):
        d = {k: v for k, v in PC["by_source_and_visible"][sname].items() if k != "all"}
        ks = sorted(d, key=int)
        ax[2].errorbar([int(k) for k in ks], [d[k]["mean"] for k in ks],
                       yerr=np.abs(np.array([d[k]["ci95"] for k in ks]).T - [d[k]["mean"] for k in ks]),
                       marker="o", capsize=2, c=c, label=sname)
    ax[2].set_xlabel("post-contact frames inside the encoded window"); ax[2].set_title("pre-contact pull, point 22")
    ax[2].legend(fontsize=7)
    for a in ax:
        a.axhline(0, c="k", lw=.5); a.axhline(1, c="k", lw=.5, ls="--")
    fig.tight_layout(); fig.savefig(FIG / "fig_contact_probe_domain.png", dpi=130); plt.close(fig)
    print("figure written")


if __name__ == "__main__":
    main()

"""COAST-faithful conceptor steering on the direction ring (arXiv 2605.17144), beside the raw chord and the spline.

What COAST does (paper section / eq.), and what each arm here keeps or changes:
  sets        success / failure activations, one mean-pooled vector per step (Sec. 3.2, A.7.1). Here: target
              condition = knot clips at the kept values the raw chord interpolates between at the held-out target
              (chord weights; no held-out clip is ever fit), failure = the same rule at the clip's source value.
  conceptor   C = R (R + a^-2 I)^-1, R = centred covariance (Eq. 1, A.9.1). Jaeger (2014) Def. 1 uses the
              UNcentred correlation E[xx']; arm family "jaegerR" uses that (sensitivity, not COAST).
  AND         COAST implementation pinv(pinv(A) + pinv(B) - I) (A.9.1 pseudocode) and Jaeger 2014 Def. 4 / Prop. 6
              Eq. 32 (basis of R(A) n R(B)); C_steer = C_t AND NOT C_s (Eq. 4 / 7).
  space       full residual (d = 1024, no truncation, A.9.4) = "full"; "pca16" / "pca64" = the same inside the top-k
              PCA of the knot clips with the complement untouched (our adaptation, not COAST).
  tokens      COAST fits on token-mean-pooled vectors and gates every token (Fig. 1, A.9.2). The gate is linear, so
              gating the 8 time tokens then mean-pooling = gating the clip mean (checked: timepool mean = meanpool
              to fp16). "tok" = conceptors fit on the 8 time tokens per clip instead of the clip mean (adaptation).
  gate        h' = h M^T, M = (1 - b) I + b C_steer on the UNcentred h (Eq. 5 / 9); identical to h + b (C h - h), so
              "multiplicative" and "additive b(Cx - x)" are the same arm. "cen" gates x - m (m = knot mean, the ring
              centre) and adds m back (adaptation). "nr" rescales the gated h to the clip's own norm (a stand-in for
              the next layer's RMSNorm that follows the hook in pi0.5; our readers read the edited vector directly).
  alpha       Stage 2 (A.10.2, Eq. 11): mean overlap tr(CsCf)/sqrt(tr(Cs^2)tr(Cf^2)) over (target, kept-source)
              pairs on the grid {0.1, 0.5, 1, 2, 10} (Table 14); keep alphas in [0.85, 0.95], else the closest.
  beta        {0.1, 0.3} retained by COAST (0.5 dropped as harmful, A.10.2) + 0.5 and 1.0 here.
  baselines   COAST's own linear/CAA arm h + a unit(mu_t - mu_s) (A.10.1, A.19) with the same conditions; positive-only
              M = (1 - b) I + b C_t (Sec. 3.2); random-eigenvector conceptor, spectrum kept (A.3).
Readers (as results/p2_bakeoff_*): ridge probe and MLP on probe folds 3-4, nearest-real R vs test clips at the target.
Each arm's endpoint edit is read at its own norm, rescaled per clip to the raw chord's ||delta|| ("chordnorm") and to
the spline's ||delta|| (the bake-off's "matched").

  python scripts/run_coast_faithful.py --layer 12 [--seed 0] [--out ...]
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_part2 as rp                                                   # noqa: E402
from wm import bakeoff as bo                                            # noqa: E402
from wm import conceptor as cc                                          # noqa: E402
from wm import geometry_checks as gc                                    # noqa: E402
from wm import manifold as mf                                           # noqa: E402
from wm.data import PROJECT_ROOT                                        # noqa: E402
from wm.p2_data import load_inputs                                      # noqa: E402
from wm.provenance import provenance                                    # noqa: E402

ALPHAS = cc.ALPHA_GRID
BETAS = (0.1, 0.3, 0.5, 1.0)
LABELS22_SEEDS = (4, 8, 9, 10)
TOL = 1e-10


def _sym_fn(M, f, tol=TOL):
    s, U = np.linalg.eigh((M + M.T) / 2)
    keep = np.abs(s) > tol * max(1.0, np.abs(s).max())
    return (U[:, keep] * f(s[keep])) @ U[:, keep].T, U[:, keep]


def pinv_sym(M):
    return _sym_fn(M, lambda s: 1.0 / s)[0]


def reduced(A_t, A_s):
    """Orthonormal basis Q [D, r] of the row spans of the two (weighted, centred or not) sample matrices."""
    A = np.vstack([A_t, A_s])
    _, s, Vt = np.linalg.svd(A, full_matrices=False)
    return Vt[s > 1e-9 * s.max()].T


def small_conceptor(a, alpha):
    """a = weighted sample matrix in Q coords [n, r]; C = R (R + alpha^-2 I)^-1 with numerically-zero eigen set to 0."""
    R = a.T @ a
    lam, U = np.linalg.eigh(R)
    lam = np.where(lam > 1e-12 * max(lam.max(), 1e-300), lam, 0.0)
    return (U * (lam / (lam + alpha ** -2))) @ U.T


def and_pinv(A, B):
    """COAST A.9.1: pinv(pinv(A) + pinv(B) - I). In the reduced basis this is exact: on the complement of span(Q)
    pinv(C_t) = 0, pinv(NOT C_s) = I, so the bracket is 0 there and so is its pinv."""
    return pinv_sym(pinv_sym(A) + pinv_sym(B) - np.eye(len(A)))


def and_jaeger(A, B, tol=1e-9):
    """Jaeger 2014 Def. 4 / Prop. 6 Eq. 32: B_int (B_int' (A^+ + B^+ - I) B_int)^-1 B_int'. R(C_t) lies inside span(Q),
    so the intersection with R(NOT C_s) does too and the reduced computation is exact."""
    r = len(A)
    Ua = _sym_fn(A, lambda s: s, tol)[1]
    Ub = _sym_fn(B, lambda s: s, tol)[1]
    P = np.vstack([np.eye(r) - Ua @ Ua.T, np.eye(r) - Ub @ Ub.T])
    _, s, Vt = np.linalg.svd(P)
    s = np.concatenate([s, np.zeros(r - len(s))])
    Bi = Vt[s < 1e-7].T
    if Bi.shape[1] == 0:
        return np.zeros((r, r))
    core = Bi.T @ (pinv_sym(A) + pinv_sym(B) - np.eye(r)) @ Bi
    return Bi @ np.linalg.inv(core) @ Bi.T


def cond_rows(Y, yk, kv, weights, tok=None, centred=True):
    """Weighted sample matrix sqrt(w) (Y - w@Y) of a condition (chord weights over kept values). tok: token rows
    [n_knot, T, k] used instead of the clip means (each clip's weight split over its T tokens)."""
    rows, ws = [], []
    for v, wv in zip(kv, weights):
        if wv <= 1e-9:
            continue
        idx = np.flatnonzero(np.isclose(yk, v))
        if tok is None:
            rows.append(Y[idx])
            ws.append(np.full(len(idx), wv / len(idx)))
        else:
            T = tok.shape[1]
            rows.append(tok[idx].reshape(-1, tok.shape[2]))
            ws.append(np.full(len(idx) * T, wv / (len(idx) * T)))
    X, w = np.concatenate(rows), np.concatenate(ws)
    w = w / w.sum()
    mu = w @ X
    Xc = X - mu if centred else X
    return np.sqrt(w)[:, None] * Xc, mu


class Space:
    """Coordinates in which conceptors are fit and gates applied: full activation (V = None) or top-k PCA of the
    knot clips (V [k, D], rows orthonormal; the off-subspace residual is untouched)."""

    def __init__(self, V=None):
        self.V = V

    def fwd(self, X):
        return X if self.V is None else X @ self.V.T

    def back(self, dY):
        return dY if self.V is None else dY @ self.V


def steer_matrix(At, As, alpha, and_mode, kind="contrastive", Q=None):
    Q = reduced(At, As) if Q is None else Q
    ct, cs = small_conceptor(At @ Q, alpha), small_conceptor(As @ Q, alpha)
    if kind == "positive":
        c = ct
    else:
        c = (and_pinv if and_mode == "pinv" else and_jaeger)(ct, np.eye(len(cs)) - cs)
    return Q, c, ct, cs


def gate_delta(Yx, Q, c, beta, centre=None):
    """Endpoint edit of the COAST gate h' = h M^T, M = (1-b) I + b Q c Q' (C_steer is 0 off span(Q)), in the space
    coordinates. centre: gate Yx - centre instead of Yx."""
    h = Yx if centre is None else Yx - centre
    Ch = (h @ Q) @ c @ Q.T
    return beta * (Ch - h)


def run(args):
    t0 = time.time()
    L = args.layer
    angle = "labels" if args.labels_angle else "unsupervised"
    d = load_inputs("direction", L)
    m = rp.build(d, 64, angle, "contiguous", args.seed, 0, "smooth")
    raw = m["raw_curve"]
    periodic = True
    probe_rows = d["role"] == "probe"
    test = np.flatnonzero(d["role"] == "test")
    evals = {"probe": mf.ProbeReadout(d["X"][probe_rows], d["y"][probe_rows], periodic),
             "mlp": bo.MLPReadout(d["X"][probe_rows], d["y"][probe_rows], periodic, args.seed)}
    near = mf.NearestRealReadout(d["X"][test], d["y"][test])
    picks = rp.pick_clips(d, m["held"], args.n_clips, args.seed)
    # the paper's interpolating periodic cubic through the RAW kept centroids (Goodfire A.3), built exactly as
    # scripts/run_endpoint_diagnosis.py builds spline_interp (our stored FITPACK smoother is flawed, REPORT 4.3)
    raw_C = {float(v): c for v, c in zip(m["cent"]["values"], m["cent"]["C"])}
    Ck = np.array([raw_C[float(v)] for v in m["curve"].values])
    ci = mf.periodic_cubic(Ck, m["curve"].values, angle=m["curve"].coords, smooth=None)
    ci.aim = m["curve"].aim
    knot = m["knot"]
    Xk = d["X"][knot].astype(np.float64)
    yk, kv = d["y"][knot], raw.values.astype(float)
    centre_full = Xk.mean(0)
    tok_all = None
    if not args.no_tokens:
        tp = np.load(PROJECT_ROOT / "artifacts" / "activations" / "direction" / "vjepa2" / "timepool.npy", mmap_mode="r")
        tok_all = np.asarray(tp[np.flatnonzero(knot), L], dtype=np.float64)          # [n_knot, 8, D]
    spaces = {"full": Space(None)}
    for k in args.pca_k:
        spaces[f"pca{k}"] = Space(gc.fit_subspace(Xk, yk, "pca", k).components)
    # variant = (space, token rows?, centred covariance?)
    variants = {"full": ("full", False, True), **{s: (s, False, True) for s in spaces if s != "full"}}
    if tok_all is not None:
        variants["full_tok"] = ("full", True, True)
    variants["full_jaegerR"] = ("full", False, False)
    if args.variants:
        variants = {k: v for k, v in variants.items() if k in args.variants or k == "full"}
    Yk = {s: sp.fwd(Xk) for s, sp in spaces.items()}
    Ytok = {"full": tok_all} if tok_all is not None else {}
    # condition weights: the kept values flanking each value along the ring (aim="arc", issue #213/#266), so no
    # condition is built from a foreign knot on wrapped arcs; the chord/spline comparison arms keep build()'s aim
    craw = dataclasses.replace(raw, aim=args.cond_aim)
    Wkept = cc.chord_weights(craw, kv)
    Wt = {float(t): w for t, w in zip(m["held"], cc.chord_weights(craw, m["held"]))}

    def rows(var, weights):
        s, tok, cen = variants[var]
        return cond_rows(Yk[s], yk, kv, weights, Ytok.get(s) if tok else None, cen)

    # ---- Stage 2 aperture selection per variant (knot clips only; all targets x every 2nd kept value as source)
    aperture = {}
    for var in variants:
        ovs = {a: [] for a in ALPHAS}
        Asrc = [rows(var, ws)[0] for ws in Wkept[::args.alpha_stride]]
        for t, wt in Wt.items():
            At, _ = rows(var, wt)
            for As in Asrc:
                Q = reduced(At, As)
                at, as_ = At @ Q, As @ Q
                for a in ALPHAS:
                    ovs[a].append(cc.overlap(small_conceptor(at, a), small_conceptor(as_, a)))
        mo = {a: float(np.mean(v)) for a, v in ovs.items()}
        sel, inb = cc.select_alpha(mo)
        inband = [a for a in ALPHAS if cc.OVERLAP_BAND[0] <= mo[a] <= cc.OVERLAP_BAND[1]]
        aperture[var] = {"mean_overlap": {str(a): v for a, v in mo.items()}, "selected": sel, "in_band": inb,
                         "in_band_alphas": inband}
    print(f"[L{L} s{args.seed}] aperture done {time.time() - t0:.0f}s", flush=True)

    # ---- steering
    rng = np.random.default_rng(args.seed + 7)
    acc, diag = {}, {v: {"trace_C_steer": [], "chord_energy_in_range": [], "cos_edit_chord": [],
                         "overlap_selected": []} for v in variants}
    ids_all, dtheta_all = [], []

    def add(arm, x, dend, tgt, refs):
        a = acc.setdefault(arm, {f"{e}_{n}": [] for e in ("probe", "mlp", "R")
                                 for n in ("own", "chordnorm", "splinenorm", "interpnorm")})
        a.setdefault("norm_ratio_to_chord", [])
        for n, ref in (("own", None), ("chordnorm", refs["chord_raw"]), ("splinenorm", refs["spline"]),
                       ("interpnorm", refs["spline_interp"])):
            dm = dend if ref is None else bo.match_norm(dend, ref)
            xs = x + dm
            for e, ev in evals.items():
                a[f"{e}_{n}"].append(mf.value_error(ev.predict(xs), tgt, periodic))
            a[f"R_{n}"].append(near.score(xs, x, tgt))
        a["norm_ratio_to_chord"].append(np.linalg.norm(dend, axis=1) / np.linalg.norm(refs["chord_raw"], axis=1))

    for tgt, pick in picks.items():
        x, src = d["X"][pick].astype(np.float64), d["y"][pick]
        ids_all.append(d["df"]["id"].to_numpy()[pick])
        dtheta_all.append(np.abs(mf.value_error(src, np.full(len(src), tgt), True)))
        deltas = bo.arm_deltas(x, src, tgt, m, None, None, None, periodic)
        refs = {k: v[0] for k, v in deltas.items()}
        refs["spline_interp"] = m["pca"].lift_delta(ci(ci.coord_of_value(np.full(len(src), tgt)))
                                                    - ci(ci.coord_of_value(src)))
        for arm in ("chord_raw", "spline_interp", "spline", "centroid_transport"):
            add(arm, x, refs[arm], tgt, refs)
        Wsrc = cc.chord_weights(craw, src)
        # per clip edits for every (variant, and_mode, kind, alpha, beta, gate)
        edits = {}
        caa = np.zeros_like(x)
        cache = {}
        for i, s in enumerate(src):
            key = float(s)
            if key not in cache:
                cache[key] = {}
                for var, (sp_name, tok, cen) in variants.items():
                    At, mu_t = rows(var, Wt[tgt])
                    As, mu_s = rows(var, Wsrc[i])
                    sel = aperture[var]["selected"]
                    alist = ALPHAS if (var in args.alpha_sweep_variants) else (sel,)
                    Q0 = reduced(At, As)
                    for a in alist:
                        for and_mode in (("jaeger", "pinv") if a == sel else ("jaeger",)):
                            Q, c, ct, cs = steer_matrix(At, As, a, and_mode, Q=Q0)
                            cache[key][(var, and_mode, "contr", a)] = (Q, c)
                            if and_mode == "jaeger" and a == sel:
                                cache[key][(var, "-", "pos", a)] = (Q, ct)
                                cache[key][(var, "diag", a)] = (Q, c, ct, cs, mu_t, mu_s)
                    if var == "full":
                        v = mu_t - mu_s          # COAST linear / CAA direction (A.19), same conditions
                        cache[key]["caa"] = v / max(np.linalg.norm(v), 1e-12)
            for ck, val in cache[key].items():
                if ck == "caa" or ck[1] == "diag":
                    continue
                var, and_mode, kind, a = ck
                sp = spaces[variants[var][0]]
                Q, c = val
                yx = sp.fwd(x[i:i + 1])
                cen_c = sp.fwd(centre_full[None])
                for b in BETAS:
                    for gname, cent in (("unc", None), ("cen", cen_c)):
                        if gname == "cen" and not (kind == "contr" and and_mode == "jaeger"):
                            continue
                        dY = gate_delta(yx, Q, c, b, cent)
                        edits.setdefault((var, and_mode, kind, a, b, gname), np.zeros_like(x))[i] = sp.back(dY)[0]
            caa[i] = cache[key]["caa"]
            # diagnostics (selected alpha, Jaeger AND, full variants)
            for var in variants:
                Q, c, ct, cs, mu_t, mu_s = cache[key][(var, "diag", aperture[var]["selected"])]
                sp = spaces[variants[var][0]]
                dg = diag[var]
                dg["trace_C_steer"].append(float(np.trace(c)))
                ch = sp.fwd(refs["chord_raw"][i:i + 1])[0]
                # energy of the chord edit inside the range of C_steer (eigvecs with mu > 0.01)
                s_, U_ = np.linalg.eigh(c)
                Ur = Q @ U_[:, s_ > 0.01]
                dg["chord_energy_in_range"].append(float(np.sum((Ur.T @ ch) ** 2) / max(ch @ ch, 1e-12)))
                e = gate_delta(sp.fwd(x[i:i + 1]), Q, c, 0.3)[0]
                dg["cos_edit_chord"].append(float(e @ ch / max(np.linalg.norm(e) * np.linalg.norm(ch), 1e-12)))
                dg["overlap_selected"].append(cc.overlap(ct, cs))
        # CAA at chord norm (COAST's linear baseline; its alpha is a magnitude, here set by the chord's ||delta||)
        add("coast_linear_caa", x, caa * np.linalg.norm(refs["chord_raw"], axis=1, keepdims=True), tgt, refs)
        for (var, and_mode, kind, a, b, gname), dend in edits.items():
            arm = f"{var}|{kind}|{and_mode}|a{a:g}|b{b:g}|{gname}"
            add(arm, x, dend, tgt, refs)
            if gname == "unc":       # norm-restored gate: h' rescaled to ||h|| (downstream-RMSNorm stand-in)
                xs = x + dend
                dnr = xs * (np.linalg.norm(x, axis=1, keepdims=True) / np.linalg.norm(xs, axis=1, keepdims=True)) - x
                if kind == "contr" and and_mode == "jaeger" and a == aperture[var]["selected"]:
                    add(f"{var}|{kind}|{and_mode}|a{a:g}|b{b:g}|unc_normrestored", x, dnr, tgt, refs)
        # random-eigenvector ablation (A.3) for the full space, selected alpha, Jaeger AND: same spectrum, random
        # orthonormal eigenvectors in R^D
        var = "full"
        a = aperture[var]["selected"]
        for b in (() if args.no_random else (0.1, 0.3, 1.0)):
            dend = np.zeros_like(x)
            for i, s in enumerate(src):
                Q, c = cache[float(s)][(var, "jaeger", "contr", a)]
                sv = np.clip(np.linalg.eigvalsh(c), 0, None)
                G = np.linalg.qr(rng.standard_normal((x.shape[1], len(sv))))[0]
                h = x[i]
                dend[i] = b * ((G * sv) @ (G.T @ h) - h)
            add(f"full|random_eigvec|jaeger|a{a:g}|b{b:g}|unc", x, dend, tgt, refs)
        print(f"[L{L} s{args.seed}] target {tgt} done {time.time() - t0:.0f}s ({len(cache)} source values)",
              flush=True)

    ids = np.concatenate(ids_all)
    dtheta = np.concatenate(dtheta_all)
    chord = {k: np.concatenate(v) for k, v in acc["chord_raw"].items()}
    interp = {k: np.concatenate(v) for k, v in acc["spline_interp"].items()}
    arms = {}
    for arm, a in acc.items():
        c = {k: np.concatenate(v) for k, v in a.items()}
        o = {k: float(np.mean(v)) for k, v in c.items()}
        o["gap_vs_chord_probe_chordnorm"] = rp.paired_bootstrap(c["probe_chordnorm"] - chord["probe_chordnorm"], ids,
                                                                seed=args.seed)
        o["gap_vs_chord_probe_own"] = rp.paired_bootstrap(c["probe_own"] - chord["probe_own"], ids, seed=args.seed)
        o["gap_vs_interp_probe_interpnorm"] = rp.paired_bootstrap(c["probe_interpnorm"] - interp["probe_interpnorm"],
                                                                  ids, seed=args.seed)
        o["gap_vs_interp_probe_own"] = rp.paired_bootstrap(c["probe_own"] - interp["probe_own"], ids, seed=args.seed)
        far = dtheta > 90
        o["probe_chordnorm_dtheta_le90"] = float(c["probe_chordnorm"][~far].mean())
        o["probe_chordnorm_dtheta_gt90"] = float(c["probe_chordnorm"][far].mean())
        arms[arm] = o
    unsteered = {e: float(np.mean([mf.value_error(ev.predict(d["X"][p]), t, periodic).mean() for t, p in picks.items()]))
                 for e, ev in evals.items()}
    dg_out = {v: {k: (float(np.mean(x_)) if len(x_) else None) for k, x_ in dg.items()}
              for v, dg in diag.items()}
    # shared-variance diagnostic: squared distance between the condition means vs total within-condition variance
    msr = []
    for t, wt in Wt.items():
        _, mut = cond_rows(Xk, yk, kv, wt)
        for ws in Wkept[::args.alpha_stride]:
            As, mus = cond_rows(Xk, yk, kv, ws)
            msr.append(float((mut - mus) @ (mut - mus)) / float(np.sum(As ** 2)))
    out = {"dataset": "direction", "layer": L, "seed": args.seed, "angle_source": m["curve"].coord_source,
           "held_out_values": m["held"].tolist(), "holdout": m["design"],
           "provenance": provenance(None, seeds={"seed": args.seed, "random_eigvec": args.seed + 7}, layer=L,
                                    pool="meanpool (+timepool for full_tok fits)", holdout="contiguous",
                                    spline="smooth", k=64),
           "n_steered": int(len(ids)), "unsteered_err": unsteered,
           "frac_pairs_dtheta_gt_90": float(np.mean(dtheta > 90)),
           "aperture": aperture, "diagnostics": dg_out,
           "mean_shift_sq_over_source_total_variance": {"mean": float(np.mean(msr)), "median": float(np.median(msr))},
           "arms": arms,
           "arm_key": "space|kind|AND|alpha|beta|gate; space: full (1024-d, COAST), pca16/pca64 (adaptation), full_tok "
                      "(fit on 8 time tokens per clip), full_jaegerR (uncentred Jaeger Def. 1 correlation); kind: contr "
                      "= C_t AND NOT C_s (Eq. 4), pos = C_t only; gate: unc = COAST h M^T on the raw h (Eq. 5/9), cen = "
                      "about the knot mean, unc_normrestored = unc then rescaled to ||h||. Metrics: {probe,mlp}_{own,"
                      "chordnorm,splinenorm,interpnorm} = mean abs angle error to target (deg), edit rescaled per clip to the "
                      "raw chord / stored smoothing spline / paper's interpolating spline ||delta||; R_* = nearest-real R (higher "
                      "better); norm_ratio_to_chord = ||edit|| / ||raw-chord edit||.",
           "config": {"alphas": list(ALPHAS), "betas": list(BETAS), "n_clips": args.n_clips,
                      "alpha_stride": args.alpha_stride, "pca_k": args.pca_k,
                      "alpha_sweep_variants": args.alpha_sweep_variants, "variants": list(variants), "cond_aim": args.cond_aim,
                      "chord_aim": raw.aim, "target_condition_knots": {str(t): {str(v): float(x_) for v, x_ in zip(kv, w)
                                                                            if x_ > 1e-9} for t, w in Wt.items()}, "tokens": not args.no_tokens},
           "runtime_s": time.time() - t0}
    path = Path(args.out) if args.out else PROJECT_ROOT / "results" / f"p2_coast_faithful_L{L}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path} in {time.time() - t0:.0f}s")
    return out


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--labels-angle", action="store_true")
    p.add_argument("--n-clips", type=int, default=48)
    p.add_argument("--alpha-stride", type=int, default=2)
    p.add_argument("--pca-k", type=int, nargs="*", default=[16, 64])
    p.add_argument("--alpha-sweep-variants", nargs="*", default=["full"])
    p.add_argument("--no-tokens", action="store_true")
    p.add_argument("--variants", nargs="*", default=None, help="subset of variants (full always kept)")
    p.add_argument("--no-random", action="store_true", help="skip the random-eigenvector ablation")
    p.add_argument("--cond-aim", default="arc", choices=("arc", "coord"))
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse())

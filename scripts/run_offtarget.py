"""Conserved-quantity off-target of the direction edits (Bao et al. 2608.23526: an edit should leave the invariant on
its level set). Speed is not an edit target here, so a direction edit should not move the speed readout; start
position likewise.

Steered states are not stored, so they are regenerated with run_part2's code path and the flags recorded in
results/p2_steer_direction_direction_L{12,22}_contiguous_rawchord.json (contiguous hold-out, smoothing spline, k 64,
K 50, unsupervised angle, seed 0, 48 test clips per target; controls: 20 random smooth curves on the first 16 clips
per target), and the probe-QR arm with run_bakeoff's refit basis (knot rows at kept values). Regeneration is
checked against stored numbers (delta norm per arm; bake-off probe-QR norm and probe error).

Arms (endpoint state, waypoint K): spline (run_part2 'manifold'), chord_smoothed (run_part2 'linear', chord through
the smoothed knots), chord_raw ('linear_raw'), probe_qr (bake-off, own norm and norm-matched to the spline),
random_curve (run_part2 'random_unmatched' controls, per clip averaged over the 20 draws).
Readouts, both disjoint from the steered test clips:
  speed    Part 1 recipe (train-standardised ridge, alpha by CV over the split's train folds) fit on SPEED-set train
           clips, read on direction-set activations at the same point (a transferred probe; its accuracy on unedited
           direction test clips is reported).
  start    the same recipe for (start_x, start_y) fit on DIRECTION-set train clips; metres x 32 px/m (render_twin).
Off-target = |readout(edited) - readout(unedited)| per clip; 95% bootstrap CI over clips (clip id is the resampling
unit). Natural spread = pooled within-label SD of the unedited readout on direction test clips (speed: groups of
equal (motion, speed, acceleration); start: RMS of the probe residual, px).

  python scripts/run_offtarget.py --layer 12
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from wm import bakeoff as bo
from wm import manifold as mf
from wm.data import PROJECT_ROOT
from wm.p2_data import load_inputs
from wm.probes import Standardizer, cv_select_alpha, fit_ridge, score
from functools import partial

_spec = importlib.util.spec_from_file_location("run_part2", Path(__file__).with_name("run_part2.py"))
p2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p2)

PX_PER_M = 32.0
ARMS = ("spline", "chord_smoothed", "chord_raw", "probe_qr", "probe_qr_norm_matched", "random_curve")
P2_NAME = {"spline": "manifold", "chord_smoothed": "linear", "chord_raw": "linear_raw"}


class RidgeReadout:
    """Part 1 probe recipe: train-standardised features, closed-form ridge, alpha by CV over the split's folds."""

    def __init__(self, X, Y, folds, kind):
        self.st = Standardizer().fit(X)
        Xs = self.st.transform(X)
        Y = np.asarray(Y, float).reshape(len(Y), -1)
        self.alpha = cv_select_alpha(Xs, Y, folds, score_fn=partial(score, kind=kind))["alpha"]
        self.W, self.b = fit_ridge(Xs, Y, self.alpha)

    def __call__(self, X):
        return self.st.transform(np.asarray(X, float)) @ self.W + self.b


def offtarget(read, x0, x1):
    """Per-clip absolute change of a readout: |r(x1) - r(x0)| (Euclidean norm over outputs)."""
    return np.linalg.norm(read(x1) - read(x0), axis=1)


def cluster_boot(v, ids, n_boot=1000, seed=0):
    """Mean of v and its 95% CI, resampling clip ids (a clip steered to several targets is one unit)."""
    v, ids = np.asarray(v, float), np.asarray(ids)
    u, inv = np.unique(ids, return_inverse=True)
    s, c = np.bincount(inv, v), np.bincount(inv)
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
        b.append((w * s).sum() / (w * c).sum())
    return {"mean": float(v.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "n_rows": int(len(v)), "n_clips": int(len(u))}


def pooled_within_sd(r, groups):
    r, groups = np.asarray(r, float).reshape(len(r), -1), np.asarray(groups)
    ss, dof = 0.0, 0
    for g in np.unique(groups):
        rg = r[groups == g]
        if len(rg) > 1:
            ss += ((rg - rg.mean(0)) ** 2).sum()
            dof += len(rg) - 1
    return float(np.sqrt(ss / max(dof, 1)))


def endpoint_states(d, m, picks, basis, std, K, n_control_clips):
    """{arm: (X_edited [n, D], X_orig [n, D], clip ids [n])}, plus regeneration checks."""
    ids = d["df"]["id"].to_numpy()
    acc = {a: ([], [], []) for a in ARMS}
    dose = {a: [] for a in P2_NAME}
    qr_norm, qr_err = [], []
    probe_rows = d["role"] == "probe"
    ev = mf.ProbeReadout(d["X"][probe_rows], d["y"][probe_rows], True)
    for tgt, pick in picks.items():
        x, src = d["X"][pick].astype(float), d["y"][pick]
        Z, resid = m["pca"].project(x), m["pca"].complement(x)
        Zarms = p2.subspace_arms(Z, src, tgt, m, K)
        for a, name in P2_NAME.items():
            W = p2.compose(m["pca"], Zarms[name], resid)
            dose[a].append(np.linalg.norm(W - x[:, None], axis=-1))
            _put(acc[a], W[:, -1], x, ids[pick])
        dq, _ = bo.probe_qr_delta(x, basis, tgt, True, *std)
        qr_norm.append(np.linalg.norm(dq, axis=1))
        qr_err.append(mf.value_error(ev.predict(x + dq), tgt, True))
        _put(acc["probe_qr"], x + dq, x, ids[pick])
        ref = p2.compose(m["pca"], Zarms["manifold"], resid)[:, -1] - x
        _put(acc["probe_qr_norm_matched"], x + bo.match_norm(dq, ref), x, ids[pick])
        sub = slice(0, n_control_clips)
        ends = [p2.compose(m["pca"], p2.curve_coords(cv, Z[sub], src[sub], tgt, K), resid[sub])[:, -1]
                for cv in m["controls"]["random_unmatched"]]
        acc["random_curve"][0].append(np.stack(ends))          # [draws, n_sub, D]
        acc["random_curve"][1].append(x[sub])
        acc["random_curve"][2].append(ids[pick][sub])
    out = {}
    for a, (xe, xo, ii) in acc.items():
        if a == "random_curve":
            out[a] = (np.concatenate(xe, axis=1), np.concatenate(xo), np.concatenate(ii))
        else:
            out[a] = (np.concatenate(xe), np.concatenate(xo), np.concatenate(ii))
    checks = {"delta_norm_per_waypoint_end": {a: float(np.concatenate(v)[:, -1].mean()) for a, v in dose.items()},
              "probe_qr_mean_norm": float(np.concatenate(qr_norm).mean()),
              "probe_qr_err_probe": float(np.concatenate(qr_err).mean())}
    return out, checks


def _put(slot, xe, xo, ii):
    slot[0].append(xe)
    slot[1].append(xo)
    slot[2].append(ii)


def arm_offtarget(states, read, scale=1.0):
    xe, xo, ii = states
    if xe.ndim == 3:        # random curves: per clip mean over draws
        v = np.mean([offtarget(read, xo, e) for e in xe], axis=0)
    else:
        v = offtarget(read, xo, xe)
    return v * scale, ii


def run(layer, K=50, n_clips=48, n_control_clips=16, seed=0, n_boot=1000):
    d = load_inputs("direction", layer, "direction")
    m = p2.build(d, 64, "unsupervised", "contiguous", seed, 20, "smooth")
    picks = p2.pick_clips(d, m["held"], n_clips, seed)
    basis, std, basis_note = bo.refit_probe_basis(d, m, "direction")
    states, checks = endpoint_states(d, m, picks, basis, std, K, n_control_clips)

    s = load_inputs("speed", layer, "speed")
    tr = s["is_train"]
    speed = RidgeReadout(s["X"][tr], s["df"]["speed_mps"].to_numpy(float)[tr], s["fold"][tr], "scalar")
    dtr = d["is_train"]
    start_xy = d["df"][["start_x", "start_y"]].to_numpy(float)
    start = RidgeReadout(d["X"][dtr], start_xy[dtr], d["fold"][dtr], "vector")
    test = d["role"] == "test"
    Xt = d["X"][test]
    df_t = d["df"][test]
    sp0 = speed(Xt)[:, 0]
    grp = (df_t["motion"] + "_" + df_t["speed_mps"].astype(str) + "_" + df_t["acceleration_mps2"].astype(str)).to_numpy()
    st_res = (start(Xt) - start_xy[test]) * PX_PER_M
    spread = {"speed_mps": pooled_within_sd(sp0, grp),
              "start_px": float(np.sqrt((st_res ** 2).sum(1).mean()))}
    vel = (df_t["motion"] == "velocity").to_numpy()
    readout_quality = {
        "speed_probe_alpha": speed.alpha,
        "speed_probe_on_direction_test_velocity_clips_mae_mps": float(np.abs(sp0[vel] - df_t["speed_mps"].to_numpy()[vel]).mean()),
        "speed_probe_on_direction_test_velocity_clips_r": float(np.corrcoef(sp0[vel], df_t["speed_mps"].to_numpy()[vel])[0, 1]),
        "speed_readout_sd_across_direction_test_clips_mps": float(sp0.std()),
        "start_probe_alpha": start.alpha,
        "start_probe_on_direction_test_rms_px": spread["start_px"],
        "start_readout_sd_across_direction_test_clips_px": float(np.sqrt(((start(Xt) - start(Xt).mean(0)) ** 2).sum(1).mean()) * PX_PER_M)}
    arms = {}
    for a in ARMS:
        res = {}
        for q, read, scale, unit in (("speed", speed, 1.0, "m/s"), ("start", start, PX_PER_M, "px")):
            v, ii = arm_offtarget(states[a], read, scale)
            b = cluster_boot(v, ii, n_boot, seed)
            sk = "speed_mps" if q == "speed" else "start_px"
            res[q] = {"unit": unit, **b, "ratio_to_natural_spread": b["mean"] / spread[sk],
                      "ratio_ci95": [c / spread[sk] for c in b["ci95"]]}
            if q == "speed":
                xe, xo, _ = states[a]
                e = xe.mean(0) if xe.ndim == 3 else xe
                res[q]["signed_mean_mps"] = float((read(e) - read(xo))[:, 0].mean())
        res["edit_norm_mean"] = float(np.mean(np.linalg.norm((states[a][0] - states[a][1]) if states[a][0].ndim == 2
                                                             else (states[a][0] - states[a][1][None]), axis=-1)))
        arms[a] = res
    return {"layer": layer, "arms": arms, "natural_spread": spread, "readout_quality": readout_quality,
            "regeneration_checks": checks, "probe_basis": basis_note, "held_out_values": m["held"].tolist(),
            "n_steered": int(sum(len(p) for p in picks.values()))}


def stored_checks(layer):
    r = json.loads((PROJECT_ROOT / "results" / f"p2_steer_direction_direction_L{layer}_contiguous_rawchord.json").read_text())
    b = json.loads((PROJECT_ROOT / "results" / f"p2_bakeoff_direction_direction_L{layer}_contiguous_rawchord.json").read_text())
    return {"delta_norm_per_waypoint_end": {a: r["delta_norm_per_waypoint"][n][-1] for a, n in P2_NAME.items()},
            "probe_qr_mean_norm": b["arms"]["probe_qr"]["mean_norm_unmatched"],
            "probe_qr_err_probe": b["arms"]["probe_qr"]["err_probe_unmatched"],
            "split_sha256": r["provenance"]["split_sha256"], "commit": r["provenance"]["commit"]}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    a = p.parse_args(argv)
    out = run(a.layer)
    st = stored_checks(a.layer)
    split_sha = hashlib.sha256((PROJECT_ROOT / "splits" / "split_v1.json").read_bytes()).hexdigest()
    new = out["regeneration_checks"]
    diffs = [abs(new["delta_norm_per_waypoint_end"][k] - v) / abs(v) for k, v in st["delta_norm_per_waypoint_end"].items()]
    diffs += [abs(new[k] - st[k]) / abs(st[k]) for k in ("probe_qr_mean_norm", "probe_qr_err_probe")]
    out["regeneration_checks"] = {"regenerated": new, "stored": st, "split_sha256_now": split_sha,
                                  "max_rel_diff": float(max(diffs)),
                                  "identical_within_1e-6": bool(max(diffs) < 1e-6 and split_sha == st["split_sha256"])}
    out["source"] = ("Bao et al., arXiv 2608.23526 (conserved invariant under edits); steered states regenerated from "
                     f"p2_steer_direction_direction_L{a.layer}_contiguous_rawchord.json flags")
    out["method"] = __doc__.split("\n\n", 1)[1].strip()
    Path(a.results_dir, f"p2_offtarget_direction_L{a.layer}.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()

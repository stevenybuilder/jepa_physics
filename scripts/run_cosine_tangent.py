"""Cosine between the Part 1 linear steering step and the Part 2 spline tangent, per waypoint (contiguous held-out arc).

For every steered (clip, target) of results/p2_steer_{variable}_{variable}_L{point}_contiguous.json (same roles, seed,
held-out values and picked clips, read from that file and checked), at each of the K waypoints of the spline walk:
  step    the Part 1 multi-probe subspace step x* - x (wm.steer.steer, basis QR of the all-train INLP sequence cut at
          the nested K, all K probes, unit/true target), mapped from train-standardised to raw activation space;
          the stored artifacts/inlp basis for this layer point if present, else rebuilt with wm.inlp.inlp as step 2 does
  tangent the spline path's unit tangent in the full activation space (np.gradient over consecutive waypoints,
          second-order at the ends; waypoints = PCA-64 spline walk lifted back plus the clip's residual)
  chord   the Part 2 straight-line arm: chord point at the target minus chord point at the source, lifted to full space
  net     the spline path's net displacement, endpoint minus start (the fairest single number for the step)
Reported per waypoint as mean +/- SE over clips (each clip's targets averaged first), the mean over waypoints, the
start, end and midpoint values, and by shift.

Writes results/p2_cosine_tangent_{variable}_L{point}_contiguous.json and figures/fig_cosine_tangent_{variable}.png.

  python scripts/run_cosine_tangent.py --variable direction --points 12 22
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wm.data import PROJECT_ROOT
from wm.p2_data import load_inputs
from wm.provenance import provenance

_spec = importlib.util.spec_from_file_location("run_part2", Path(__file__).with_name("run_part2.py"))
p2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p2)


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def path_cosines(W, step, chord):
    """W [n, K, D] spline waypoints, step [n, D] linear steering step, chord [n, D]. Returns per-clip cosines:
    step_tangent, tangent_chord, tangent_net [n, K]; step_chord, step_net [n]."""
    T = _unit(np.gradient(np.asarray(W, float), axis=1, edge_order=2))
    s, c, net = _unit(np.asarray(step, float)), _unit(np.asarray(chord, float)), _unit(W[:, -1] - W[:, 0])
    return {"step_tangent": np.einsum("nkd,nd->nk", T, s), "tangent_chord": np.einsum("nkd,nd->nk", T, c),
            "tangent_net": np.einsum("nkd,nd->nk", T, net),
            "step_chord": (s * c).sum(-1), "step_net": (s * net).sum(-1)}


def midpoint(a):
    """Value at the path midpoint along the last axis (mean of the two central waypoints when K is even)."""
    K = a.shape[-1]
    return a[..., K // 2] if K % 2 else 0.5 * (a[..., K // 2 - 1] + a[..., K // 2])


def clip_means(a, ids):
    """Average the rows of each clip (a clip is steered to several targets): [n_clips, ...]."""
    u, inv = np.unique(ids, return_inverse=True)
    return np.stack([a[inv == i].mean(0) for i in range(len(u))])


def mean_se(a):
    return {"mean": float(a.mean()), "se": float(a.std(ddof=1) / np.sqrt(len(a)))}


def summarise(cos, ids):
    out = {}
    for q, a in cos.items():
        c = clip_means(a, ids)
        if a.ndim == 2:
            out[q] = {"per_waypoint_mean": c.mean(0).round(5).tolist(),
                      "per_waypoint_se": (c.std(0, ddof=1) / np.sqrt(len(c))).round(5).tolist(),
                      "mean_over_waypoints": mean_se(c.mean(1)), "start": mean_se(c[:, 0]), "end": mean_se(c[:, -1]),
                      "midpoint": mean_se(midpoint(c))}
        else:
            out[q] = mean_se(c)
    return out


def part1_basis(dataset, variable, point, inlp_dir=None):
    """The Part 1 steering basis and its standardiser: stored all-train INLP .npz if present, else rebuilt exactly as
    wm.inlp.run_inlp does (step-1 all-train standardiser, step-1 alpha, nested K; no test read)."""
    from wm.inlp import basis_path, inlp, load_basis
    from wm.probes import Standardizer, layer_data, layer_matrix, load_sweep
    Y, kind, score_fn, tr, te, folds, acts, _ = layer_data(dataset, variable, "meanpool", "vjepa2", None)
    X = layer_matrix(acts, point)
    st = Standardizer().fit(X[tr])
    path = basis_path(dataset, variable, point, "meanpool", inlp_dir)
    if path.exists():
        probes = load_basis(path)
        note = {"source": "stored", "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    else:
        alpha = load_sweep(dataset, variable, "meanpool")["layers"][point]["alpha"]
        _, Q, Wp, b = inlp(st.transform(X[tr]), Y[tr], None, None, folds, alpha, score_fn, kind, protocols=("nested",))
        probes = {"Q": Q, "W": Wp, "b": b, "alpha": float(alpha), "point": point, "kind": kind}
        note = {"source": "rebuilt with wm.inlp.inlp (nested protocol, all train folds, step-1 alpha)",
                "missing_path": str(path)}
    note.update({"K_probes": int(len(probes["W"])), "alpha": float(probes["alpha"]), "rank": int(probes["W"].shape[0] *
                                                                                                 probes["W"].shape[2])})
    return probes, (st.mean, st.std), note


def part1_step(x, probes, target, periodic, mu, sd):
    """Part 1 steer at all K probes (the nested cut), raw-space delta x* - x."""
    from wm.steer import build_basis, encode, steer
    V = build_basis(list(probes["W"]))
    Xs = (x - mu) / sd
    t = encode(np.full(len(x), target), "circular" if periodic else "scalar")
    return (steer(Xs, V, probes, t, len(probes["W"])) - Xs) * sd


def run_point(variable, point, args):
    ref_path = Path(args.ref_dir) / f"p2_steer_{variable}_{variable}_L{point}_contiguous.json"
    ref = json.loads(ref_path.read_text())
    rp = ref["provenance"]
    seed, K, k, spline = rp["seeds"]["seed"], ref["K"], ref["k"], rp["spline"]
    d = load_inputs(variable, point, variable)
    periodic = d["periodic"]
    m = p2.build(d, k, "unsupervised", "contiguous", seed, 0, spline)
    assert np.allclose(m["held"], ref["held_out_values"]), (m["held"], ref["held_out_values"])
    assert m["curve"].coord_source == ref["angle_source"], (m["curve"].coord_source, ref["angle_source"])
    n_clips = int(next(iter(ref["n_steered_per_target"].values())))
    picks = p2.pick_clips(d, m["held"], n_clips, seed)
    assert {str(t): len(p) for t, p in picks.items()} == ref["n_steered_per_target"]
    ref_ids = sorted((r["id"], r["target"]) for r in ref["rows"] if r["arm"] == "manifold")
    probes, (mu, sd), basis_note = part1_basis(variable, variable, point, args.inlp_dir)

    acc, ids, shifts = {}, [], []
    for tgt, pick in picks.items():
        x, src = d["X"][pick].astype(float), d["y"][pick]
        Z, resid = m["pca"].project(x), m["pca"].complement(x)
        arms = p2.subspace_arms(Z, src, tgt, m, K)
        W, Wl = p2.compose(m["pca"], arms["manifold"], resid), p2.compose(m["pca"], arms["linear"], resid)
        cos = path_cosines(W, part1_step(x, probes, tgt, periodic, mu, sd), Wl[:, -1] - Wl[:, 0])
        for q, a in cos.items():
            acc.setdefault(q, []).append(a)
        ids.append(d["df"]["id"].to_numpy()[pick])
        shifts.append(p2.shift_of(src, tgt, periodic))
    cat = {q: np.concatenate(v) for q, v in acc.items()}
    ids, shifts = np.concatenate(ids), np.concatenate(shifts)
    tgts = np.concatenate([np.full(len(p), t) for t, p in picks.items()])
    assert sorted(zip(ids.tolist(), tgts.tolist())) == ref_ids, "steered (clip, target) rows differ from the reference"

    edges = (np.array([0, 45, 90, 135, 180.01]) if periodic
             else np.quantile(shifts, np.linspace(0, 1, 5)) + np.r_[np.zeros(4), 1e-9])
    by_shift = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (shifts >= lo) & (shifts < hi)
        if sel.sum() > 2 and len(np.unique(ids[sel])) > 2:
            s = summarise({q: a[sel] for q, a in cat.items()}, ids[sel])
            by_shift.append({"shift_lo": float(lo), "shift_hi": float(hi), "n_rows": int(sel.sum()),
                             **{q: ({kk: v for kk, v in s[q].items() if not kk.startswith("per_waypoint")}
                                    if isinstance(s[q], dict) and "midpoint" in s[q] else s[q]) for q in s}})
    out = {"dataset": variable, "variable": variable, "layer": point, "holdout": ref["holdout"],
           "held_out_values": ref["held_out_values"],
           "provenance": provenance(None, seeds={"seed": seed}, layer=point, pool="meanpool", holdout="contiguous",
                                    spline=spline, k=k, K=K, reference_result=ref_path.name,
                                    reference_commit=rp["commit"]),
           "part1_basis": basis_note,
           "angle_source": m["curve"].coord_source, "n_rows": int(len(ids)), "n_clips": int(len(np.unique(ids))),
           "definitions": {
               "step": "Part 1 multi-probe subspace step x* - x (wm.steer.steer, V = QR of the all-train INLP probe "
                       "sequence cut at the nested K, all K probes, target encode(value) required of every probe), "
                       "computed in train-standardised coordinates and mapped back to raw activations (times SD)",
               "tangent": "unit tangent of the spline path in full activation space: np.gradient along the K "
                          "waypoints (central differences, second-order one-sided at the ends) of the PCA-64 spline "
                          "walk lifted back plus the clip's residual",
               "chord": "Part 2 linear arm direction: polyline (chord) point at the target minus at the source, lifted",
               "net": "spline path endpoint minus start",
               "midpoint": "mean of waypoints K/2-1 and K/2 (K even)",
               "se": "SE over unique clips; each clip's (clip, target) rows averaged first"},
           "same_rows_as_reference": True, "cosines": summarise(cat, ids), "by_shift": by_shift}
    out["split_sha256_matches_reference"] = out["provenance"]["split_sha256"] == rp["split_sha256"]
    assert out["split_sha256_matches_reference"], "split file differs from the reference run's"
    c = out["cosines"]
    print(f"{variable} L{point} (Part 1 K={basis_note['K_probes']}, {basis_note['source'].split(' ')[0]}): "
          f"step.tangent start {c['step_tangent']['start']['mean']:.3f} mid {c['step_tangent']['midpoint']['mean']:.3f} "
          f"end {c['step_tangent']['end']['mean']:.3f} mean {c['step_tangent']['mean_over_waypoints']['mean']:.3f}; "
          f"step.chord {c['step_chord']['mean']:.3f}; step.net {c['step_net']['mean']:.3f}; tangent.chord start "
          f"{c['tangent_chord']['start']['mean']:.3f} mid {c['tangent_chord']['midpoint']['mean']:.3f} end "
          f"{c['tangent_chord']['end']['mean']:.3f}")
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.results_dir) / f"p2_cosine_tangent_{variable}_L{point}_contiguous.json").write_text(
        json.dumps(out, indent=1))
    return out


def plot(outs, path, variable):
    fig, axes = plt.subplots(1, len(outs), figsize=(4.6 * len(outs), 3.6), sharey=True, squeeze=False)
    for ax, o in zip(axes[0], outs):
        c = o["cosines"]
        K = len(c["step_tangent"]["per_waypoint_mean"])
        wp = np.arange(K)
        for q, label in (("step_tangent", "Part 1 step . spline tangent"), ("tangent_chord", "spline tangent . chord")):
            mu, se = np.array(c[q]["per_waypoint_mean"]), np.array(c[q]["per_waypoint_se"])
            line, = ax.plot(wp, mu, label=label)
            ax.fill_between(wp, mu - se, mu + se, color=line.get_color(), alpha=0.25, lw=0)
        sc = c["step_chord"]
        ax.axhline(sc["mean"], color="C2", label="Part 1 step . chord")
        ax.axhspan(sc["mean"] - sc["se"], sc["mean"] + sc["se"], color="C2", alpha=0.25, lw=0)
        ax.set(xlabel="waypoint index", title=f"layer {o['layer']} (Part 1 K = {o['part1_basis']['K_probes']}, "
                                              f"n = {o['n_clips']} clips)")
    axes[0][0].set_ylabel("cosine (mean +/- SE over clips)")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"{variable}: Part 1 linear step vs spline tangent, contiguous held-out arc", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--variable", required=True, choices=("direction", "speed", "acceleration"))
    p.add_argument("--points", type=int, nargs="+", required=True)
    p.add_argument("--inlp-dir", default=None)
    p.add_argument("--ref-dir", default=str(PROJECT_ROOT / "results"), help="where the p2_steer_*_contiguous files are")
    p.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--figures-dir", default=str(PROJECT_ROOT / "figures"))
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse()
    outs = [run_point(a.variable, pt, a) for pt in a.points]
    Path(a.figures_dir).mkdir(parents=True, exist_ok=True)
    plot(outs, Path(a.figures_dir) / f"fig_cosine_tangent_{a.variable}.png", a.variable)

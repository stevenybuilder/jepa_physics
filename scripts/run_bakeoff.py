"""Steering bake-off at matched delivered norm: held-out angle error vs edit rank (wm.bakeoff).

Same roles, held-out design and steered clips as run_part2.py. The probe-QR arm's basis is refit here on knot
rows at kept values (wm.inlp), unless --probe-basis gives a stored one (flagged: that one saw the evaluators' probe
folds and the held-out values).

Writes results/p2_bakeoff_{dataset}_{variable}_L{layer}_{holdout}.json and figures/fig4_bakeoff_rank_*.png.

  python scripts/run_bakeoff.py --dataset direction --layer 12 --holdout contiguous
"""
import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wm import bakeoff as bo
from wm import manifold as mf
from wm.data import PROJECT_ROOT
from wm.p2_data import load_inputs
from wm.provenance import layer_role, provenance

_spec = importlib.util.spec_from_file_location("run_part2", Path(__file__).with_name("run_part2.py"))
p2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p2)


def find_basis(args, variable, d, m):
    """Default: refit the probe sequence on knot rows at kept values (bo.refit_probe_basis). --probe-basis PATH uses
    a stored Part 1 basis instead (fit on all train folds and values: includes the evaluators' probe folds and the
    held-out values, so that arm is favoured; flagged)."""
    if args.probe_basis:
        from wm.inlp import load_basis
        return load_basis(args.probe_basis), None, {"rows": f"stored basis {args.probe_basis}: all train folds and "
                                                              "values (includes probe folds + held-out values; "
                                                              "favours probe_qr)"}
    if args.no_probe_qr:
        return None, None, {"rows": "skipped (--no-probe-qr)"}
    return bo.refit_probe_basis(d, m, variable, args.max_probe_rounds)


def run(args):
    variable = args.variable or args.dataset
    d = load_inputs(args.dataset, args.layer, variable, args.act_dir, args.table, args.split)
    periodic = d["periodic"]
    m = p2.build(d, args.k, "labels" if args.labels_angle else "unsupervised", args.holdout, args.seed, 0, args.spline)
    picks = p2.pick_clips(d, m["held"], args.n_clips, args.seed)
    basis, basis_std, basis_note = find_basis(args, variable, d, m)
    res = bo.run_bakeoff(d, m, picks, periodic, basis, args.seed, basis_std)
    out = {"dataset": args.dataset, "variable": variable, "layer": args.layer, "holdout": m["design"],
           **layer_role(args.dataset, args.layer, variable),
           "provenance": provenance(args.split, seeds={"seed": args.seed}, layer=args.layer, pool="meanpool",
                                    holdout=args.holdout, spline=args.spline, k=args.k),
           "spline": m["curve"].kind, "k": int(m["pca"].components.shape[0]), "probe_basis": basis_note,
           "norm_matching": "every arm's edit rescaled per clip to the spline arm's ||delta|| ('matched'); "
                            "'unmatched' = each arm's own norm",
           "evaluators": {"probe": "ridge on probe folds (shares filter geometry with probe_qr)",
                          "mlp": "one-hidden-layer MLP (64) on probe folds"},
           **res}
    tag = f"{args.dataset}_{variable}_L{args.layer}_{args.holdout}"
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.results_dir) / f"p2_bakeoff_{tag}.json").write_text(json.dumps(out, indent=1))
    plot(out, Path(args.figures_dir) / f"fig4_bakeoff_rank_{tag}.png", tag, periodic)
    return out


def plot(out, path, tag, periodic):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for e, marker in (("probe", "o"), ("mlp", "s")):
        for i, (arm, r) in enumerate(out["arms"].items()):
            ax.scatter(r["effective_rank"], r[f"err_{e}_matched"], marker=marker, color=f"C{i}",
                       label=arm if e == "probe" else None)
        ax.axhline(out["unsteered_err"][e], color="0.6", lw=0.8, ls="--" if e == "probe" else ":")
    ax.set_xscale("log")
    ax.set(xlabel="effective edit rank (participation ratio of deltas)",
           ylabel=f"held-out error to target ({'degrees' if periodic else 'label units'})",
           title=f"Matched-norm bake-off, {tag}\ncircle: ridge probe, square: MLP; dashed/dotted: unsteered")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=("direction", "speed", "acceleration"))
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--variable", default=None)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--holdout", default="contiguous", choices=mf.HOLDOUT_DESIGNS)
    p.add_argument("--spline", default="interp", choices=("interp", "smooth"))
    p.add_argument("--n-clips", type=int, default=48)
    p.add_argument("--labels-angle", action="store_true")
    p.add_argument("--probe-basis", default=None,
                   help="use a stored step-2 basis (flagged: fit on all train) instead of refitting on knot rows")
    p.add_argument("--no-probe-qr", action="store_true")
    p.add_argument("--max-probe-rounds", type=int, default=None, help="cap on INLP rounds for the refit")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--act-dir", default=None)
    p.add_argument("--table", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--figures-dir", default=str(PROJECT_ROOT / "figures"))
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse())

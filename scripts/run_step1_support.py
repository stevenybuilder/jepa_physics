"""Step 1 support runs on stored activations (spec 5.1 (a)-(c)); Part 1 extras, not the paper's protocol.

  python scripts/run_step1_support.py [--only onset transfer spatial] [--pools meanpool diskpool]

Writes results/p1a_support_onset_{dataset}_{pool}.json (Cartesian vs polar onset, constant-speed clips),
results/p1a_support_transfer_{pool}.json (direction probe from the direction set read on the speed and
acceleration sets) and results/p1a_support_spatial_{pool}.json (start x < 0 vs x > 0). Missing
activation files are skipped with a message.
"""
import argparse
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import load_table  # noqa: E402
from wm.probes import RESULTS, all_nan_clips, layer_matrix, load_activations, split_rows, targets, write_json  # noqa: E402
from wm.support import onset_compare, spatial_curve, transfer_curve  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--only", nargs="+", default=["onset", "transfer", "spatial"],
                    choices=["onset", "transfer", "spatial"])
parser.add_argument("--pools", nargs="+", default=["meanpool", "diskpool"], choices=["meanpool", "diskpool"])
parser.add_argument("--n-boot", type=int, default=200)
parser.add_argument("--act-root", type=Path, default=None)
parser.add_argument("--results", type=Path, default=None)
args = parser.parse_args()
results = args.results or RESULTS


def layer_getter(dataset, pool):
    """X [N, d] by layer point (8-step pools averaged over time), or None if the file is missing. The
    getter's .valid is False for clips NaN at every step (disk out of frame throughout): excluded, not imputed."""
    try:
        acts = load_activations(dataset, pool, act_root=args.act_root)
    except FileNotFoundError as e:
        print(f"skip {dataset}/{pool}: {e}")
        return None
    get = lru_cache(maxsize=2)(lambda p: layer_matrix(acts, p))
    get.valid = ~all_nan_clips(acts)
    return get


SOURCE = "stored activations; start_x and motion from load_table (metadata.json start_position_xy_m, motion)"

for pool in args.pools:
    if "onset" in args.only:   # (a) constant-speed clips: direction set's velocity half, and the speed set
        for dataset, motion in (("direction", "velocity"), ("speed", "velocity")):
            get_X = layer_getter(dataset, pool)
            if get_X is None:
                continue
            df = load_table(dataset)
            tr, _, folds = split_rows(dataset, df)
            keep = (df["motion"].to_numpy()[tr] == motion) & get_X.valid[tr]
            Yd, _, sd = targets(df, "direction")
            Yv, _, sv = targets(df, "vxvy")
            res = onset_compare(get_X, {"vxvy": (Yv, sv), "sincos": (Yd, sd)}, tr[keep], folds[keep], args.n_boot)
            res.update({"dataset": dataset, "pool": pool, "clips": f"train clips with motion == {motion}",
                        "label": "extra (spec 5.1 a): Cartesian (vx, vy) vs polar (sin, cos) onset", "source": SOURCE,
                        "n_all_nan_excluded": int((~get_X.valid[tr]).sum())})
            av = {k: v["availability"] for k, v in res["targets"].items()}
            print(f"onset {dataset}/{pool}: vxvy {av['vxvy']['onset']} {av['vxvy']['onset_ci']}, "
                  f"sincos {av['sincos']['onset']} {av['sincos']['onset_ci']}, diff CI {res['onset_difference']['ci95']}")
            write_json(results / f"p1a_support_onset_{dataset}_{pool}.json", res)

    if "transfer" in args.only:   # (b) direction probe from the direction set, read on the other sets
        get_src = layer_getter("direction", pool)
        if get_src is not None:
            df = load_table("direction")
            tr, _, folds = split_rows("direction", df)
            tr, folds = tr[get_src.valid[tr]], folds[get_src.valid[tr]]
            Y, _, _ = targets(df, "direction")
            sets = {}
            for dataset in ("speed", "acceleration"):
                get_t = layer_getter(dataset, pool)
                if get_t is None:
                    continue
                dft = load_table(dataset)
                ok = get_t.valid
                subsets = None
                if dataset == "speed":   # the direction set's constant speeds are 1-7 m/s
                    sp = dft["speed_mps"].to_numpy()[ok]
                    subsets = {"speed_below_1": sp < 1.0, "speed_1_to_4": sp >= 1.0}
                sets[dataset] = (lambda p, g=get_t, k=ok: g(p)[k], targets(dft, "direction")[0][ok], subsets)
            if sets:
                rows = transfer_curve(get_src, Y, tr, folds, sets)
                write_json(results / f"p1a_support_transfer_{pool}.json", {
                    "pool": pool, "source": "direction set, train clips (probe fit + CV alpha)",
                    "targets": {k: f"all {len(v[1])} {k}-set clips (clips NaN at every step excluded)" for k, v in sets.items()},
                    "label": "extra (spec 5.1 b): direction transfer; source score is CV (no test read)",
                    "space": "direction-set train-standardised coordinates", "layers": rows})
                print(f"transfer/{pool}: written ({', '.join(sets)})")

    if "spatial" in args.only:   # (c) start x < 0 vs x > 0, train clips of the direction set
        get_X = layer_getter("direction", pool)
        if get_X is not None:
            df = load_table("direction")
            tr, _, folds = split_rows("direction", df)
            Y, _, _ = targets(df, "direction")
            tr, folds = tr[get_X.valid[tr]], folds[get_X.valid[tr]]
            x = df["start_x"].to_numpy()[tr]
            neg, pos = x < 0, x > 0
            rows = spatial_curve(get_X, Y, tr[neg], tr[pos], folds[neg], folds[pos])
            write_json(results / f"p1a_support_spatial_{pool}.json", {
                "pool": pool, "clips": "direction set train clips, split by start x sign (x == 0 dropped; clips NaN at every step excluded)",
                "n_neg": int(neg.sum()), "n_pos": int(pos.sum()), "source": SOURCE,
                "label": "extra (spec 5.1 c): spatial generalisation; within = CV on the fitting side",
                "layers": rows})
            print(f"spatial/{pool}: written (n_neg={int(neg.sum())}, n_pos={int(pos.sum())})")

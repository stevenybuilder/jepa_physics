"""Step 2: iterative nullspace probing at the step-1 peak and onset layers (or --layer), with the random-removal
control; --all-layers instead runs the sequence at every layer for the dimensionality-vs-depth plot.

  python scripts/run_step2.py --dataset direction [--variable direction] [--layer 12] [--all-layers]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.inlp import run_dims_vs_layer, run_inlp  # noqa: E402
from wm.probes import VARIABLES, chosen_layers, load_sweep  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=list(VARIABLES), default=None)
parser.add_argument("--variable", default=None, help="default: the dataset's primary variable")
parser.add_argument("--layer", type=int, default=None, help="layer point 0..25; overrides --layer-role")
parser.add_argument("--layer-role", default="both", choices=["peak", "onset", "both"],
                    help="step-1 CV peak, availability onset (the paper's emergence layer), or both (default)")
parser.add_argument("--pool", default="meanpool", choices=["meanpool", "timepool", "diskpool"])
parser.add_argument("--all-layers", action="store_true")
parser.add_argument("--seeds", type=int, default=10)
parser.add_argument("--act-root", type=Path, default=None)
parser.add_argument("--results", type=Path, default=None)
parser.add_argument("--inlp-dir", type=Path, default=None)
args = parser.parse_args()

for dataset in [args.dataset] if args.dataset else VARIABLES:
    variable = args.variable or dataset
    if args.all_layers:
        run_dims_vs_layer(dataset, variable, args.pool, args.act_root, args.results, args.inlp_dir)
    else:
        points = [args.layer] if args.layer is not None else list(
            chosen_layers(load_sweep(dataset, variable, args.pool, args.results), args.layer_role))
        for point in points:
            run_inlp(dataset, variable, point, args.pool, args.seeds, True, args.act_root, args.results,
                     args.inlp_dir)

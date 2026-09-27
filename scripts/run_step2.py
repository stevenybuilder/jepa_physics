"""Step 2: iterative nullspace probing at the step-1 peak and onset layers and the paper's layer 8 (or --layer), with
the random-removal control, under the nested (CV-faithful) and paper (test-scored) protocols; --all-layers instead
runs the sequence at every layer for the dimensionality-vs-depth plot. --recipe adam runs the paper's C.11 Adam probe
sequence instead (paper protocol; writes ..._L{L}_adam.json).

  python scripts/run_step2.py --dataset direction [--variable direction] [--layer 12] [--all-layers]
                              [--protocol both|nested|paper] [--recipe ridge|adam]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.inlp import run_adam_sequence, run_dims_vs_layer, run_inlp  # noqa: E402
from wm.probes import VARIABLES, chosen_layers, load_sweep  # noqa: E402
from wm.provenance import PAPER_LAYER  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=list(VARIABLES), default=None)
parser.add_argument("--variable", default=None, help="default: the dataset's primary variable")
parser.add_argument("--layer", type=int, default=None, help="layer point 0..25; overrides --layer-role")
parser.add_argument("--layer-role", default="all", choices=["peak", "onset", "both", "paper", "all"],
                    help="step-1 CV peak, availability onset, both, the paper's layer 8, or all three (default)")
parser.add_argument("--protocol", default="both", choices=["both", "nested", "paper"],
                    help="nested = CV-faithful K (always computed); paper adds the test-scored curve (reads test)")
parser.add_argument("--recipe", default="ridge", choices=["ridge", "adam"],
                    help="adam = the paper's C.11 Adam probe at every round (paper protocol)")
parser.add_argument("--adam-batch", type=int, default=None, help="Adam minibatch size (default: full batch)")
parser.add_argument("--adam-max-rounds", type=int, default=None)
parser.add_argument("--pool", default="meanpool", choices=["meanpool", "timepool", "diskpool"])
parser.add_argument("--all-layers", action="store_true")
parser.add_argument("--model", default="vjepa2", choices=["vjepa2", "random"],
                    help="random = the random-init ViT-L control; outputs carry _random")
parser.add_argument("--seeds", type=int, default=10)
parser.add_argument("--act-root", type=Path, default=None)
parser.add_argument("--results", type=Path, default=None)
parser.add_argument("--inlp-dir", type=Path, default=None)
args = parser.parse_args()
protocols = ("nested", "paper") if args.protocol != "nested" else ("nested",)


def layer_set(sweep, role):
    """{point: role}: chosen_layers for peak/onset/both, plus the paper's layer 8 for 'paper' and 'all'."""
    out = {} if role == "paper" else dict(chosen_layers(sweep, "both" if role == "all" else role))
    if role in ("paper", "all"):
        out[PAPER_LAYER] = f"{out[PAPER_LAYER]}+paper_layer" if PAPER_LAYER in out else "paper_layer"
    return out


for dataset in [args.dataset] if args.dataset else VARIABLES:
    variable = args.variable or dataset
    if args.all_layers:
        run_dims_vs_layer(dataset, variable, args.pool, args.act_root, args.results, args.inlp_dir, args.model,
                          protocols)
    else:
        points = [args.layer] if args.layer is not None else list(
            layer_set(load_sweep(dataset, variable, args.pool, args.results, args.model), args.layer_role))
        for point in points:
            if args.recipe == "adam":
                run_adam_sequence(dataset, variable, point, args.pool, args.act_root, args.results, args.model,
                                  args.adam_batch, args.adam_max_rounds)
            else:
                run_inlp(dataset, variable, point, args.pool, args.seeds, True, args.act_root, args.results,
                         args.inlp_dir, args.model, protocols)

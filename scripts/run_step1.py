"""Step 1: layer-wise probing. Writes results/p1a_{dataset}_{variable}_{pool}[_shuffled].json.

  python scripts/run_step1.py --dataset direction [--variable direction] [--pool meanpool] [--shuffled]
Without --dataset/--variable, runs every variable of every dataset.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.probes import VARIABLES, layer_sweep  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=list(VARIABLES), default=None)
parser.add_argument("--variable", default=None)
parser.add_argument("--pool", default="meanpool", choices=["meanpool", "timepool", "diskpool"])
parser.add_argument("--model", default="vjepa2", choices=["vjepa2", "random", "videomae"])
parser.add_argument("--shuffled", action="store_true", help="control: permute train labels")
parser.add_argument("--act-root", type=Path, default=None)
parser.add_argument("--results", type=Path, default=None)
args = parser.parse_args()

for dataset in [args.dataset] if args.dataset else VARIABLES:
    for variable in [args.variable] if args.variable else VARIABLES[dataset]:
        layer_sweep(dataset, variable, args.pool, args.model, args.shuffled, args.act_root, args.results)

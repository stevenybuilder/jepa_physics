"""Extract pooled V-JEPA 2 activations.

  python scripts/extract.py run   --dataset speed [--model vjepa2|random] [--limit N | --ids 1 2 3]
  python scripts/extract.py merge --dataset speed [--model vjepa2|random]

Output: artifacts/activations/{dataset}/{model}/ (or --out).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import DATASETS, PROJECT_ROOT, load_table  # noqa: E402
from wm.extract import merge, run_extraction  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("command", choices=["run", "merge"])
parser.add_argument("--dataset", required=True, choices=DATASETS)
parser.add_argument("--model", default="vjepa2", choices=["vjepa2", "random"])
parser.add_argument("--out", type=Path, default=None)
parser.add_argument("--limit", type=int, default=None, help="first N clips in manifest order")
parser.add_argument("--ids", type=int, nargs="+", default=None)
parser.add_argument("--batch-size", type=int, default=8)
args = parser.parse_args()

out_dir = args.out or PROJECT_ROOT / "artifacts" / "activations" / args.dataset / args.model
df = load_table(args.dataset)
if args.command == "run":
    if args.ids is not None:
        missing = set(args.ids) - set(df["id"])
        assert not missing, f"ids not in manifest: {sorted(missing)}"
        df = df[df["id"].isin(args.ids)]
    if args.limit is not None:
        df = df.head(args.limit)
    run_extraction(df, args.model, out_dir, batch_size=args.batch_size)
else:
    merge(out_dir, df["id"].tolist())

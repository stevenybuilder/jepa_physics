"""Spec §5.1 parity check: the paper's Adam + weight-decay probe (App. B grid) vs our closed-form ridge,
once per variable at one layer (default: the step-1 CV peak). Writes results/p1a_probe_recipe_check.json.

  python scripts/check_probe_recipe.py [--variables direction speed acceleration] [--layer-role peak|onset]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.adam_probe import LRS, WDS, cv_adam, parity  # noqa: E402
from wm.probes import (ALPHAS, RESULTS, cv_select_alpha, layer_fraction, load_activations, load_sweep,  # noqa: E402
                       load_table, split_rows, standardized_layer, targets, write_json)

parser = argparse.ArgumentParser()
parser.add_argument("--variables", nargs="+", default=["direction", "speed", "acceleration"])
parser.add_argument("--layer", type=int, default=None)
parser.add_argument("--layer-role", default="peak", choices=["peak", "onset"])
parser.add_argument("--epochs", type=int, default=None, help="default: 100 direction, 50 scalars (paper C.11)")
parser.add_argument("--act-root", type=Path, default=None)
parser.add_argument("--results", type=Path, default=None)
args = parser.parse_args()

out = {"rule": "parity if |Adam CV R2 - ridge CV R2| <= max(fold SD); both on the same 5 train folds and "
               "train-standardised features", "grid": {"lr": list(LRS), "weight_decay": list(WDS)}, "variables": {}}
for variable in args.variables:
    df = load_table(variable)
    point = args.layer if args.layer is not None else load_sweep(variable, variable, "meanpool", args.results)[
        "availability"][args.layer_role]
    Y, kind, score_fn = targets(df, variable)
    tr, te, folds = split_rows(variable, df)
    Xtr, _ = standardized_layer(load_activations(variable, act_root=args.act_root), point, tr, te)
    ridge = cv_select_alpha(Xtr, Y[tr], folds, ALPHAS, score_fn)
    ridge = {k: ridge[k] for k in ("alpha", "cv_mean", "cv_sd", "cv_mae_mean", "cv_mae_sd")}
    epochs = args.epochs or (100 if kind == "circular" else 50)
    adam = cv_adam(Xtr, Y[tr], folds, score_fn, epochs)
    out["variables"][variable] = {"point": point, "frac": layer_fraction(point), "ridge": ridge, "adam": adam,
                                  "parity": parity(ridge, adam)}
    print(f"{variable} point {point}: ridge R2 {ridge['cv_mean']:.3f}±{ridge['cv_sd']:.3f} "
          f"adam R2 {adam['cv_mean']:.3f}±{adam['cv_sd']:.3f} (lr={adam['lr']}, wd={adam['wd']}) "
          f"within={out['variables'][variable]['parity']['within']}")
write_json(Path(args.results or RESULTS) / "p1a_probe_recipe_check.json", out)

"""Pixel + disk-trajectory baselines (spec §6 item 2). Writes results/p1a_{dataset}_{variable}_{pixels,trajectory}.json.

  python scripts/run_pixel_baseline.py [--dataset direction] [--workers 8]
First run decodes every clip once into artifacts/pixels/{dataset}.npy (32² grayscale) + _centroid.npy.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.pixels import load_pixels, run_baseline  # noqa: E402
from wm.probes import VARIABLES  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(VARIABLES), default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for dataset in [args.dataset] if args.dataset else VARIABLES:
        gray, cen = load_pixels(dataset, args.workers)
        for variable in VARIABLES[dataset]:
            for fs in ("pixels", "trajectory"):
                r = run_baseline(dataset, variable, fs, gray, cen)["layers"][0]
                acc = f" acc15={r['cv_acc15_mean']:.3f}" if "cv_acc15_mean" in r else ""
                print(f"{dataset}/{variable} {fs:10s} alpha={r['alpha']:.3g} cv R2={r['cv_mean']:.3f}±{r['cv_sd']:.3f} "
                      f"cv MAE={r['cv_mae_mean']:.4g} test R2={r['test_r2']:.3f} test MAE={r['test_mae']:.4g}{acc}",
                      flush=True)

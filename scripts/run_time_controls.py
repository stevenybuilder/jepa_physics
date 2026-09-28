"""Experiment A controls: is the speed-invariant clock a learned code or architecture (positional information)?

Runs the clock-vs-odometer test (run_time_manifold.clock_layer), the t / distance decoding and the geometry on the
RANDOM-INIT encoder's speed-set timepool (extracted on CPU for this control; same torch.manual_seed(0) init as every
other random-init array), side by side with V-JEPA 2 on the same set, and states the odometer prediction as an explicit
ratio: odometer => (fast slope / slow slope) = mean fast speed / mean slow speed of the held-out clips (acceleration
set, from rest: sqrt of the acceleration ratio). Static-disk control: the supplied sets have no zero-speed clips
(speed set 0.25-4 m/s; direction set velocity clips 1-7 m/s, accelerating clips start at rest but accelerate at
>= 2 m/s^2; acceleration set >= 0.25 m/s^2), so it is not run.

Writes results/p5_time_manifold_controls.json (does not touch p5_time_manifold.json).

  python scripts/run_time_controls.py --act-root /dev/shm/wm_p5 --jobs 8
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from wm.data import PROJECT_ROOT
from wm.provenance import provenance

_spec = importlib.util.spec_from_file_location("run_time_manifold", Path(__file__).with_name("run_time_manifold.py"))
tmr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tmr)


def band_ratio(dataset):
    m = tmr.meta(dataset)
    band = tmr.band_of(m, dataset)
    held = (m["role"] != "knot") & (band != "")
    x = m["accel"] if dataset == "acceleration" else m["speed"]
    fast, slow = x[held & (band == "fast")].mean(), x[held & (band == "slow")].mean()
    out = {"variable": "acceleration_mps2" if dataset == "acceleration" else "speed_mps",
           "mean_slow": float(slow), "mean_fast": float(fast), "ratio_fast_over_slow": float(fast / slow)}
    out["odometer_slope_ratio"] = float(np.sqrt(fast / slow)) if dataset == "acceleration" else float(fast / slow)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-root", default=str(PROJECT_ROOT / "artifacts" / "activations"))
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 4, 8, 12, 19, 22])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "p5_time_manifold_controls.json"))
    args = ap.parse_args()
    t0 = time.time()
    rand = Path(args.act_root) / "speed" / "random" / "timepool.npy"
    jobs = [("clock", l, delayed(tmr.clock_layer)(rand, l, "speed")) for l in args.layers]
    jobs += [("decode", l, delayed(tmr.decode_layer)(rand, l, "speed")) for l in args.layers]
    jobs += [("geometry", l, delayed(tmr.geometry_layer)(rand, l, "speed")) for l in args.layers]
    outs = Parallel(n_jobs=args.jobs, verbose=5)(j[2] for j in jobs)
    res = {"speed/random/timepool": {"clock": {}, "decode": {}, "geometry": {}}}
    for (sec, l, _), (_, o) in zip(jobs, outs):
        res["speed/random/timepool"][sec][str(l)] = o
    res["odometer_prediction_as_ratio"] = {ds: band_ratio(ds) for ds in ("speed", "acceleration", "direction")}
    res["static_disk_control"] = ("not run: no zero-speed clips exist in the supplied sets (speed 0.25-4 m/s; direction-set "
                                  "velocity clips 1-7 m/s, accelerating clips start at rest but reach >= 2 m/s^2 * tau; "
                                  "acceleration set >= 0.25 m/s^2); rendering + extracting a static set needs the GPU")
    res["keys"] = {"speed/random/timepool.{clock,decode,geometry}[point]": "same fields as p5_time_manifold.json "
                   "clock / decode / geometry, on the random-init encoder's speed-set timepool (artifacts extracted on CPU "
                   "with scripts/extract.py run --model random --store-timepool; meanpool rows checked against "
                   "artifacts/activations/speed/random/meanpool.npy)",
                   "odometer_prediction_as_ratio": "held-out (non-knot) clips per band; an odometer predicts fast slope / "
                   "slow slope = odometer_slope_ratio (= mean speed ratio; sqrt of the acceleration ratio for the "
                   "acceleration set, from rest); a clock predicts 1"}
    res["provenance"] = provenance(seeds={"bootstrap": 0, "random_init": "torch.manual_seed(0)"}, act_root=args.act_root,
                                   layers=args.layers, wall_s=round(time.time() - t0, 1))
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("wrote", args.out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

"""End-to-end run of scripts/run_geometry_checks.py and scripts/run_part2.py on a tiny fake meanpool."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
N_LAYERS, D, PER_VALUE = 3, 32, 8


def fake_dataset(tmp, dataset):
    """64 values x 8 clips; layer 1 holds a planted ring (direction) or line (speed); a split with 5 train folds."""
    rng = np.random.default_rng(0)
    if dataset == "direction":
        y = np.repeat(np.arange(64) * 360.0 / 64, PER_VALUE)
        r = np.radians(y)
        signal = np.zeros((len(y), D))
        signal[:, 0], signal[:, 1] = 5 * np.cos(r), 5 * np.sin(r)
        table = {"theta_degrees": y, "speed_mps": 1.0, "acceleration_mps2": 0.0,
                 "motion": np.where(np.arange(len(y)) % 2, "velocity", "acceleration")}
    else:
        y = np.repeat(np.linspace(0.25, 4.0, 64), PER_VALUE)
        signal = np.zeros((len(y), D))
        signal[:, 0] = 3 * y
        table = {"theta_degrees": 0.0, "speed_mps": y, "acceleration_mps2": 0.0, "motion": "velocity"}
    n = len(y)
    ids = np.arange(n)
    X = rng.standard_normal((n, N_LAYERS, D)).astype(np.float32) * 0.3
    X[:, 1] += signal
    act = tmp / "act"
    act.mkdir()
    np.save(act / "meanpool.npy", X)
    (act / "ids.json").write_text(json.dumps(ids.tolist()))
    pd.DataFrame({"id": ids, "label": y, **table}).to_csv(tmp / "table.csv", index=False)
    # stratified: in each value, 2 of 8 clips are test, the other 6 cycle through folds 0..4
    fold = {}
    for v in np.unique(y):
        members = ids[y == v]
        for j, i in enumerate(members):
            fold[str(i)] = -1 if j < 2 else (j + int(v * 10)) % 5
    (tmp / "split.json").write_text(json.dumps({"fold": fold}))
    return tmp


def run_script(name, tmp, dataset, *extra):
    """Run a script's parse + run in-process with command-line arguments (same code path as the CLI)."""
    spec = importlib.util.spec_from_file_location(name[:-3], ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = ["--dataset", dataset, "--layer", "1", "--k", "16",
            "--act-dir", str(tmp / "act"), "--table", str(tmp / "table.csv"), "--split", str(tmp / "split.json"),
            "--results-dir", str(tmp / "results"), "--figures-dir", str(tmp / "figures"), *extra]
    module.run(module.parse(argv))


@pytest.mark.parametrize("dataset", ["direction", "speed"])
def test_scripts_end_to_end(tmp_path, dataset):
    tmp = fake_dataset(tmp_path, dataset)
    run_script("run_geometry_checks.py", tmp, dataset)
    geo = json.loads((tmp / "results" / f"p2_geometry_{dataset}_L1.json").read_text())
    assert (tmp / "figures" / f"fig4_centroid_plane_{dataset}_L1.png").exists()
    assert set(geo["loo"]) == {"1", "2", "4", "8", "16"}
    if dataset == "direction":
        assert geo["angle"]["source"].startswith("unsupervised")
        assert geo["loo"]["16"]["fp32"]["winner"] == "cubic"
    else:
        assert geo["knot_spacing"]["better"] == "linear"

    run_script("run_part2.py", tmp, dataset, "--variable", dataset, "--n-clips", "12", "--K", "6")
    tag = f"{dataset}_{dataset}_L1"
    res = json.loads((tmp / "results" / f"p2_steer_{tag}.json").read_text())
    for fig in ("gap_vs_shift", "path_energy"):
        assert (tmp / "figures" / f"fig4_{fig}_{tag}.png").exists()
    assert len(res["held_out_values"]) == 16
    s = res["summary"]
    # the real steer beats the shuffled control
    assert s["manifold"]["overall"]["nearest_real_R"] > s["shuffled_control"]["overall"]["nearest_real_R"]
    if dataset == "direction":
        far_lin = s["linear"]["by_shift"][-1]["excess_to_curve"]
        far_man = s["manifold"]["by_shift"][-1]["excess_to_curve"]
        assert far_lin > 1.0 and far_lin > 3 * abs(far_man)   # large shifts: the line leaves the ring

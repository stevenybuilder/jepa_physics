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

    run_script("run_part2.py", tmp, dataset, "--variable", dataset, "--n-clips", "12", "--K", "11")
    tag = f"{dataset}_{dataset}_L1_scattered"
    res = json.loads((tmp / "results" / f"p2_steer_{tag}.json").read_text())
    for fig in ("gap_vs_shift", "path_energy"):
        assert (tmp / "figures" / f"fig4_{fig}_{tag}.png").exists()
    assert len(res["held_out_values"]) == 16 and res["holdout"]["design"] == "scattered"
    s = res["summary"]
    # the real steer beats every shuffled-centroid draw
    shuf = res["controls"]["shuffled_control"]["nearest_real_R"]
    assert shuf["n_draws"] == 20 and shuf["spline_rank"] == 1
    assert s["manifold"]["overall"]["nearest_real_R"] > shuf["p95"]
    assert res["K"] == 11 and "behaviour_manifold" in res and "deviations_from_goodfire" in res
    if dataset == "direction":
        far_lin = s["linear"]["by_shift"][-1]["excess_to_curve"]
        far_man = s["manifold"]["by_shift"][-1]["excess_to_curve"]
        assert far_lin > 1.0 and far_lin > 3 * abs(far_man)   # large shifts: the line leaves the ring
        wp = res["waypoint_readout"]
        lin, man = wp["linear"][-1], wp["manifold"][-1]      # the bin nearest 180 degrees
        assert len(lin["radius"]) == 11 and min(lin["radius"]) < 0.5 * min(man["radius"])   # radius collapse
        assert s["linear"]["by_shift"][-1]["behaviour_energy"] > 1.3 * s["manifold"]["by_shift"][-1]["behaviour_energy"]
        assert res["isometry_behaviour"] > 0.9 and res["isometry_probe"] > 0.9
        assert "manifold_transport" in s


@pytest.mark.parametrize("dataset,design", [("direction", "contiguous"), ("speed", "contiguous"),
                                            ("speed", "extrapolation")])
def test_part2_holdout_designs(tmp_path, dataset, design):
    tmp = fake_dataset(tmp_path, dataset)
    run_script("run_part2.py", tmp, dataset, "--variable", dataset, "--n-clips", "6", "--K", "5",
               "--holdout", design, "--n-controls", "3", "--goodfire-baseline", "--spline", "smooth", "--no-bf16")
    res = json.loads((tmp / "results" / f"p2_steer_{dataset}_{dataset}_L1_{design}.json").read_text())
    assert res["holdout"]["design"] == design and len(res["held_out_values"]) == 8
    assert res["spline"] == "smoothing"
    assert len(res["sagitta_per_target"]) == 8 and all(r["sagitta"] >= 0 for r in res["sagitta_per_target"])
    assert set(res["goodfire_comparison"]["summary"]) == {"goodfire_linear", "goodfire_manifold"}
    assert res["controls"]["random_control"]["nearest_real_R"]["n_draws"] == 3
    if design == "extrapolation":
        assert res["held_out_values"] == sorted(res["held_out_values"])[-8:]


def test_part2_heldout_context(tmp_path):
    """Direction spline from the direction set, steering clips of a second dataset with the same ring code."""
    tmp = fake_dataset(tmp_path, "direction")
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    fake_dataset(ctx, "direction")
    ids = json.loads((ctx / "act" / "ids.json").read_text())
    X = np.load(ctx / "act" / "meanpool.npy")
    X = X + 0.05 * np.random.default_rng(1).standard_normal(X.shape).astype(np.float32)
    np.save(ctx / "act" / "meanpool.npy", X)
    run_script("run_part2.py", tmp, "direction", "--variable", "direction", "--n-clips", "6", "--K", "5",
               "--n-controls", "2", "--no-bf16", "--context-dataset", "speed",
               "--context-act-dir", str(ctx / "act"), "--context-table", str(ctx / "table.csv"),
               "--context-split", str(ctx / "split.json"))
    res = json.loads((tmp / "results" / "p2_steer_direction_direction_L1_scattered_ctx-speed.json").read_text())
    assert res["context"]["steered_dataset"] == "speed" and len(ids) == len(X)
    man = res["summary"]["manifold"]["overall"]
    assert man["nearest_real_R_context"] > 0.3 and man["nearest_real_R"] > 0.3


def test_load_pair_checks_alignment(tmp_path):
    from wm.p2_data import load_pair
    tmp = fake_dataset(tmp_path, "direction")
    kw = dict(act_dirs=(tmp / "act", tmp / "act"), tables=(tmp / "table.csv", tmp / "table.csv"),
              splits=(tmp / "split.json", tmp / "split.json"))
    a, b = load_pair("direction", "speed", 1, "direction", **kw)
    assert a["X"].shape == b["X"].shape and b["periodic"]
    (tmp / "act" / "ids.json").write_text(json.dumps(list(range(1, 513))))
    with pytest.raises(AssertionError):
        load_pair("direction", "speed", 1, "direction", **kw)

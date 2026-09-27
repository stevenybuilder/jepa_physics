"""Fake meanpool.npy (real split and labels, 26 x 32 dims, planted signal) -> steps 1-3 + figures."""
import json
import runpy
import sys
from pathlib import Path

from fake_acts import write_fake_meanpool

ROOT = Path(__file__).resolve().parents[1]


def run(script, *args):
    """Run a CLI script in this process (as `python scripts/<script> args`), saving the import cost."""
    argv = sys.argv
    sys.argv = [script, *args]
    try:
        runpy.run_path(str(ROOT / "scripts" / script), run_name="__main__")
    finally:
        sys.argv = argv


def test_pipeline_on_fake_activations(tmp_path):
    acts, results, inlp_dir, figs = tmp_path / "acts", tmp_path / "results", tmp_path / "inlp", tmp_path / "figs"
    for dataset in ("direction", "speed"):
        write_fake_meanpool(acts, dataset)
    common = ["--act-root", str(acts), "--results", str(results)]
    for dataset in ("direction", "speed"):
        run("run_step1.py", "--dataset", dataset, "--variable", dataset, *common)
        run("run_step2.py", "--dataset", dataset, "--inlp-dir", str(inlp_dir), "--seeds", "2", *common)
        run("run_step3.py", "--dataset", dataset, "--inlp-dir", str(inlp_dir), *common)
    run("run_step2.py", "--dataset", "direction", "--all-layers", "--inlp-dir", str(inlp_dir), *common)
    run("run_step2_angles.py", "--inlp-dir", str(inlp_dir), *common)
    run("make_figures.py", "--results", str(results), "--figures", str(figs))

    sweep = json.loads((results / "p1a_direction_direction_meanpool.json").read_text())
    assert len(sweep["layers"]) == 26 and sweep["n_train"] + sweep["n_test"] == 1500
    av = sweep["availability"]
    assert 5 <= av["onset"] <= 14 and av["onset_ci"][0] <= av["onset"] <= av["onset_ci"][1]
    assert set(sweep["layers"][0]["by_motion"]) == {"acceleration", "velocity"}
    peak = av["peak"]
    assert (results / f"p1b_direction_direction_meanpool_L{peak}.json").exists()
    assert (inlp_dir / f"direction_direction_L{peak}.npz").exists()
    steer = json.loads((results / f"p1c_direction_L{peak}.json").read_text())
    assert steer["single"][-1]["mae_to_target"] < steer["single"][0]["mae_to_target"]
    assert "test_radius" in sweep["layers"][peak] and str(peak) in sweep["test_radius"]
    assert steer["eval_probe"]["out_of_fold"]["r2_fold_mean"] <= steer["eval_probe"]["in_sample"]["r2"]
    assert "radius" in steer["single"][-1] and steer["radius_matched"]["single"][-1]["n"] == steer["K"]
    assert len(steer["random_nulls"]["rows"]) == steer["K"] and steer["random_nulls"]["n_draws"] >= 20
    angles = json.loads((results / "step2_subspace_angles.json").read_text())
    rows = [r for r in angles["pairs"]["direction_vs_speed"] if "mapped" in r]
    assert rows and all(0 <= r["mapped"]["min_angle_deg"] <= 90 for r in rows)
    for name in ("fig2d_subspace_angles", "fig1_layer_curves", "fig1b_direction_circle", "fig2_inlp", "fig2b_dim_vs_layer",
                 "fig3_steering", "fig3b_shift_heatmap", "fig3c_steering_nulls"):
        assert (figs / f"{name}.png").exists(), name

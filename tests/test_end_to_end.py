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
        run("run_step3.py", "--dataset", dataset, "--inlp-dir", str(inlp_dir), "--strict-eval", *common)
    run("run_step2.py", "--dataset", "direction", "--all-layers", "--inlp-dir", str(inlp_dir), *common)
    for batch in ("64", "0"):
        run("run_step2.py", "--dataset", "direction", "--recipe", "adam", "--layer", "9", "--adam-batch", batch,
            "--adam-max-rounds", "6", *common)
    run("run_step2_angles.py", "--inlp-dir", str(inlp_dir), *common)
    run("make_figures.py", "--results", str(results), "--figures", str(figs))

    sweep = json.loads((results / "p1a_direction_direction_meanpool.json").read_text())
    assert len(sweep["layers"]) == 26 and sweep["n_train"] + sweep["n_test"] == 1500
    prov = sweep["provenance"]
    assert len(prov["split_sha256"]) == 64 and prov["git_commit"] and prov["pool"] == "meanpool" and prov["timestamp_utc"]
    assert json.loads((results / f"p1c_direction_L{json.loads((results / 'p1a_direction_direction_meanpool.json').read_text())['availability']['peak']}.json").read_text())["provenance"]["layer"] is not None
    av = sweep["availability"]
    assert 5 <= av["onset"] <= 14 and av["onset_ci"][0] <= av["onset"] <= av["onset_ci"][1]
    assert set(sweep["layers"][0]["by_motion"]) == {"acceleration", "velocity"}
    peak = av["peak"]
    assert (results / f"p1b_direction_direction_meanpool_L{peak}.json").exists()
    assert (inlp_dir / f"direction_direction_L{peak}.npz").exists()
    onset = av["onset"]
    p8 = json.loads((results / "p1b_direction_direction_meanpool_L9.json").read_text())   # paper layer, default set
    assert p8["is_paper_layer"] and p8["protocol"] == "nested" and p8["paper"]["protocol"] == "paper"
    assert "L+1" in p8["layer_note"] and json.loads((results / "p1b_direction_direction_meanpool_L8.json")
                                                     .read_text())["is_paper_layer_alt"]
    assert (results / "p1c_direction_L9.json").exists() and not (results / "p1c_direction_L8.json").exists()
    assert p8["K_probes"] == p8["K"] and p8["dims_2K"] == 2 * p8["K"] and "test" in p8["provenance"]["test_read"]
    adam = json.loads((results / "p1b_direction_direction_meanpool_L9_adam_b64.json").read_text())
    full = json.loads((results / "p1b_direction_direction_meanpool_L9_adam_full.json").read_text())
    assert adam["recipe"]["batch"] == 64 and full["recipe"]["full_batch"] and adam["recipe"]["lr"] == 1e-3 and adam["rounds"] and "dips_vs_failed" in adam["sawtooth"]
    assert onset != peak and (results / f"p1b_direction_direction_meanpool_L{onset}.json").exists()
    steer_onset = json.loads((results / f"p1c_direction_L{onset}.json").read_text())
    assert steer_onset["layer_role"] == "onset" and steer_onset["is_onset"] and not steer_onset["is_peak"]
    steer = json.loads((results / f"p1c_direction_L{peak}.json").read_text())
    assert steer["single"][-1]["mae_to_target"] < steer["single"][0]["mae_to_target"]
    assert "test_radius" in sweep["layers"][peak] and str(peak) in sweep["test_radius"]
    assert steer["eval_probe"]["out_of_fold"]["r2_fold_mean"] <= steer["eval_probe"]["in_sample"]["r2"]
    assert "radius" in steer["single"][-1] and steer["radius_matched"]["single"][-1]["n"] == steer["K"]
    assert len(steer["random_nulls"]["rows"]) == steer["K"] and steer["random_nulls"]["n_draws"] >= 20
    assert steer["off_target_probe"]["variable"] == "speed" and "off_target" in steer["single"][-1]
    assert steer["protocol"].startswith("paper protocol") and steer["strict_eval"]["n_A"] + steer["strict_eval"]["n_B"] == steer["n_test"]
    assert steer["strict_eval"]["A_to_B"]["single"][-1]["mae_to_target"] < steer["strict_eval"]["A_to_B"]["single"][0]["mae_to_target"]
    angles = json.loads((results / "step2_subspace_angles.json").read_text())
    rows = [r for r in angles["pairs"]["direction_vs_speed"] if "mapped" in r]
    assert rows and all(0 <= r["mapped"]["min_angle_deg"] <= 90 for r in rows)
    for name in ("fig2d_subspace_angles", "fig3_steering_onset", "fig2_inlp_onset", "fig1_layer_curves", "fig1b_direction_circle", "fig2_inlp", "fig2_inlp_peak", "fig2b_dim_vs_layer",
                 "fig3_steering_paper", "fig3b_shift_heatmap_paper", "fig3c_steering_nulls_paper",
                 "fig3_steering", "fig3b_shift_heatmap", "fig3c_steering_nulls"):
        assert (figs / f"{name}.png").exists(), name


def test_random_model_control_steps_2_3(tmp_path):
    acts, results, inlp_dir = tmp_path / "acts", tmp_path / "results", tmp_path / "inlp"
    path = write_fake_meanpool(acts, "direction", seed=1)
    (acts / "direction" / "random").mkdir()
    path.rename(acts / "direction" / "random" / "meanpool.npy")    # the random-init ViT-L slot
    common = ["--dataset", "direction", "--model", "random", "--act-root", str(acts), "--results", str(results)]
    run("run_step1.py", *common)
    run("run_step2.py", "--inlp-dir", str(inlp_dir), "--seeds", "2", "--layer-role", "peak", *common)
    run("run_step3.py", "--inlp-dir", str(inlp_dir), "--layer-role", "peak", "--null-draws", "20", *common)
    peak = json.loads((results / "p1a_direction_direction_meanpool_random.json").read_text())["availability"]["peak"]
    assert (inlp_dir / f"direction_direction_L{peak}_random.npz").exists()
    inl = json.loads((results / f"p1b_direction_direction_meanpool_random_L{peak}.json").read_text())
    steer = json.loads((results / f"p1c_direction_L{peak}_random.json").read_text())
    assert inl["model"] == "random" and steer["model"] == "random"
    assert not (results / f"p1c_direction_L{peak}.json").exists()
    # post-hoc onsets: the V-JEPA sweep picks up the random-init sweep's out-of-fold predictions
    write_fake_meanpool(acts, "direction", seed=0)
    run("run_step1.py", "--dataset", "direction", "--act-root", str(acts), "--results", str(results))
    av = json.loads((results / "p1a_direction_direction_meanpool.json").read_text())["availability"]
    assert "selectivity_diff_ci" in av and len(av["selectivity_diff_ci"]) == 25
    assert av["precision_onset"] is not None and "POST-HOC" in av["precision_rule"]


def test_diskpool_all_nan_clips_are_excluded_not_imputed(tmp_path):
    from fake_acts import write_fake_diskpool
    from wm.probes import split_rows
    acts, results = tmp_path / "acts", tmp_path / "results"
    tr, te, _ = split_rows("direction")
    nan_rows = list(tr[:7]) + list(te[:3])
    write_fake_diskpool(acts, "direction", nan_rows)
    run("run_step1.py", "--dataset", "direction", "--pool", "diskpool", "--act-root", str(acts), "--results", str(results))
    out = json.loads((results / "p1a_direction_direction_diskpool.json").read_text())
    info = out["all_nan_clips"]
    assert info["n_all_nan_train"] == 7 and info["n_all_nan_test"] == 3 and info["per_layer_test"][12] == 3
    assert out["n_train"] == len(tr) - 7 and out["n_test"] == len(te) - 3
    assert len(out["test_theta"]) == len(te) - 3


def test_shuffled_label_control_is_at_chance(tmp_path):
    acts, results = tmp_path / "acts", tmp_path / "results"
    write_fake_meanpool(acts, "direction")
    run("run_step1.py", "--dataset", "direction", "--shuffled", "--act-root", str(acts), "--results", str(results))
    out = json.loads((results / "p1a_direction_direction_meanpool_shuffled.json").read_text())
    assert out["shuffled"] and max(r["cv_mean"] for r in out["layers"]) < 0.05
    assert max(r["test_r2"] for r in out["layers"]) < 0.05 and min(r["test_mae"] for r in out["layers"]) > 70
    assert "selectivity_onset" not in out["availability"]


def test_paper_role_ignores_stale_point8_flag(tmp_path):
    """A p1b file from before the point-9 convention says is_paper_layer at point 8; fig2 must take point 9."""
    mf = runpy.run_path(str(ROOT / "scripts" / "make_figures.py"))
    for point, flag in ((8, True), (9, True)):
        (tmp_path / f"p1b_speed_speed_meanpool_L{point}.json").write_text(
            json.dumps({"point": point, "is_paper_layer": flag}))
    assert mf["load"](tmp_path, r"p1b_speed_speed_meanpool_L\d+\.json", "paper")["point"] == 9
    (tmp_path / "p1b_speed_speed_meanpool_L8_adam_b64.json").write_text(json.dumps({"point": 8, "is_paper_layer": True}))
    alt = mf["load_adam"](tmp_path, "speed")    # no point-9 Adam file: the point-8 one is the labelled alt fallback
    assert set(alt) == {"b64"} and alt["b64"]["point"] == 8


def test_adam_row_falls_back_to_point_8(tmp_path):
    mf = runpy.run_path(str(ROOT / "scripts" / "make_figures.py"), run_name="make_figures")
    for point, tag in ((8, "b64"), (8, "full")):
        (tmp_path / f"p1b_direction_direction_meanpool_L{point}_adam_{tag}.json").write_text(json.dumps({"point": point}))
    runs = mf["load_adam"](tmp_path, "direction")
    assert set(runs) == {"b64", "full"} and all(r["point"] == 8 for r in runs.values())
    (tmp_path / "p1b_direction_direction_meanpool_L9_adam_b64.json").write_text(json.dumps({"point": 9}))
    runs = mf["load_adam"](tmp_path, "direction")
    assert set(runs) == {"b64"} and runs["b64"]["point"] == 9          # point 9 preferred when present

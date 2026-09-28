import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


pp = _load("p1a_perpatch")
rh = _load("render_hard_stimuli")


def test_hard_seed_config_seed0_is_original_and_seeds_differ():
    name0, l0, c0 = rh.hard_seed_config(0)
    assert (name0, l0, c0) == ("hard", 0, rh.HARD)
    name1, l1, c1 = rh.hard_seed_config(1)
    assert name1 == "hard_seed1" and c1["texture_seed"] == 2
    assert {k: v for k, v in c1.items() if k != "texture_seed"} == {k: v for k, v in rh.HARD.items() if k != "texture_seed"}
    assert not np.allclose(rh.layout(l0)[1], rh.layout(l1)[1])
    assert len(rh.layout(l1)[0]) == 392


def test_onset_table_on_subgrid_matches_subcurve():
    points = [1, 2, 4, 8, 9]
    curve = [0.1, 0.95, 0.5, 0.55, 1.0]
    boots = [{"c": curve}] * 3
    full = pp.onset_table(points, {"c": curve}, boots)["c"]
    sub = pp.onset_table(points, {"c": curve}, boots, keep=[0, 2, 3, 4])["c"]
    assert full["onset"] == 2 and sub["onset"] == 9 and sub["ci95"] == [9, 9]
    assert (sub["from_point"], sub["to_point"]) == (8, 9) and np.isclose(sub["largest_jump"], 0.45)


def test_combine_seeds_mean_sd_onsets_jumps():
    def res(pts, a, onset, extra=False):
        c = {m: [a + 0.1 * i for i in range(len(pts))] for m in pp.SEED_METRICS}
        on = {m: {"onset": onset, "ci95": [onset, onset]} for m in pp.SEED_METRICS}
        r = {"points": pts, "curves": c, "onsets": on,
             "layers": [{"point": p, "pooled": {"pooled_r2_all_samples": a}} for p in pts]}
        if extra:
            r["onsets_stim_grid"] = {m: {"onset": 9, "ci95": [9, 9]} for m in pp.SEED_METRICS}
        return r
    out = pp.combine_seeds({0: res([4, 8, 9], 0.0, 9), 1: res([4, 5, 8, 9], 0.2, 5, extra=True), 2: res([4, 8, 9], 0.4, 9)})
    assert out["common_points"] == [4, 8, 9] and out["extra_points"]["1"] == [5]
    row9 = out["by_point"][2]["perpos_mean_r2"]
    assert np.isclose(row9["per_seed"]["1"], 0.5) and np.isclose(row9["mean"], np.mean([0.2, 0.5, 0.6]))
    assert np.isclose(row9["sd"], np.std([0.2, 0.5, 0.6], ddof=1))
    assert out["onsets"]["1"]["cross_half_r2"]["onset"] == 9 and out["onsets"]["1"]["all_sampled_points"]["cross_half_r2"]["onset"] == 5
    assert np.isclose(out["jump_8_to_9"]["0"]["cross_half_r2"], 0.1) and np.isclose(out["jump_8_to_9"]["mean"]["perpos_mean_r2"], 0.1)

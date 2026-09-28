"""New helpers from the paper-first QA pass: motion_rows, cartesian_angle_agreement, split_record,
result_provenance(split_path), rewrite_split_provenance."""
import json

import numpy as np

from wm import provenance as pv
from wm.paperscale import motion_rows
from wm.support import cartesian_angle_agreement


def test_motion_rows_keeps_only_velocity_in_order():
    motion = np.array(["velocity", "acceleration", "velocity", "velocity", "acceleration"])
    assert motion_rows(np.array([4, 3, 1, 0]), motion).tolist() == [3, 0]


def test_cartesian_angle_agreement_ignores_magnitude():
    rng = np.random.default_rng(0)
    th = rng.uniform(0, 2 * np.pi, 200)
    Y = np.stack([np.sin(th), np.cos(th)], 1)
    speed = rng.uniform(1, 7, 200)
    P_cart = speed[:, None] * np.stack([np.cos(th), np.sin(th)], 1)      # exact up to magnitude
    P_dir = 0.5 * Y + 0.3 * rng.standard_normal(Y.shape)                 # noisy, shrunk direct probe
    a = cartesian_angle_agreement(P_cart, P_dir, Y)
    assert a["cartesian_angle_vs_truth"]["mae_deg"] < 1e-6 and abs(a["cartesian_angle_vs_truth"]["r2"] - 1) < 1e-9
    assert a["direct_probe_vs_truth"]["mae_deg"] > 5
    assert abs(a["cartesian_angle_vs_direct_probe"]["mae_deg"] - a["direct_probe_vs_truth"]["mae_deg"]) < 1e-9


def test_result_provenance_records_the_split_it_is_given(tmp_path):
    default = pv.result_provenance({"pool": "meanpool"})
    assert default["split_file"] == "splits/split_v1.json"
    assert default["split_sha256"] == pv.sha256_file(pv.SPLIT_PATH)
    other = pv.result_provenance({"pool": "meanpool"}, pv.PROJECT_ROOT / "splits/split_hard.json")
    assert other["split_file"] == "splits/split_hard.json"
    assert other["split_sha256"] == pv.sha256_file(pv.PROJECT_ROOT / "splits/split_hard.json")
    assert pv.split_record("splits/split_hard.json") == (other["split_file"], other["split_sha256"])


def test_rewrite_split_provenance_changes_only_two_values(tmp_path):
    obj = {"n_train": 313, "layers": [{"x": 0.1234567890123}], "split_file": "not-provenance",
           "provenance": pv.result_provenance({"pool": "meanpool"})}
    f = tmp_path / "r.json"
    before = json.dumps(obj, indent=1)
    f.write_text(before)
    old, new = pv.rewrite_split_provenance(f, "splits/split_hard.json")
    after = f.read_text()
    assert old[0] == "splits/split_v1.json" and new[0] == "splits/split_hard.json"
    expect = {**obj, "provenance": {**obj["provenance"], "split_file": new[0], "split_sha256": new[1]}}
    assert after == json.dumps(expect, indent=1)
    assert json.loads(after)["split_file"] == "not-provenance"

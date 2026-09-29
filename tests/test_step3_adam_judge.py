"""Shape of results/p1c_direction_L9_adam_judge.json (scripts/run_step3_adam_judge.py, QA #475)."""
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "results" / "p1c_direction_L9_adam_judge.json"
JUDGES = {"ridge_alpha100", "ridge_alpha_1e-3", "adam_c11", "split_half"}


@pytest.fixture(scope="module")
def res():
    if not PATH.exists():
        pytest.skip("results file not generated")
    return json.loads(PATH.read_text())


def test_shape(res):
    assert set(res["table"]) == {"ridge", "adam"} == set(res["bases"])
    assert res["bases"]["ridge"]["K"] == 37 and res["bases"]["adam"]["K"] == 84
    for b, rows in res["table"].items():
        assert set(rows) == JUDGES
        K = res["bases"][b]["K"]
        for j, r in rows.items():
            assert len(res["bases"][b]["judges"][j]["curve"]) == K + 1
            for n in ("5", "18"):
                lo, hi = r[f"mae_n{n}_ci95"]
                assert lo <= r[f"mae_n{n}"] <= hi
    for key in ("commit", "dirty", "timestamp_utc", "split_sha256", "seeds"):
        assert key in res["provenance"]
    assert set(res["adam_judge_rank_matched_null"]["rows"]) == {"ridge", "adam"}


def test_n_to_10deg_integers_in_range(res):
    for rows in res["table"].values():
        for r in rows.values():
            for key in ("n_to_10deg", "n_to_10deg_sustained"):
                v = r[key]
                assert isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 37, (key, v)
                lo, hi = r[f"{key}_ci95"]
                assert isinstance(lo, int) and isinstance(hi, int) and 1 <= lo <= hi <= 37


def test_stored_judge_reproduces_headline(res):
    t = res["table"]
    assert t["ridge"]["ridge_alpha100"]["n_to_10deg"] == 5 and t["adam"]["ridge_alpha100"]["n_to_10deg"] == 18

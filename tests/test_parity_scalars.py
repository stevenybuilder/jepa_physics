"""summarize_parity_scalars.aggregate_blocks on synthetic run_part2-shaped gaps."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("sps", ROOT / "scripts" / "summarize_parity_scalars.py")
sps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sps)


def fake(seed, err_gap, ci, raw_gap):
    g = lambda m, lo, hi: {"mean": m, "ci95": [lo, hi]}
    return (seed, {"holdout": {"block_first_value": seed, "block_last_value": seed + 1.0}, "probe_test_error": 0.1,
                   "verdict": {"call": "x"},
                   "gaps": {"manifold_minus_linear": {"probe_err_to_target": g(err_gap, *ci),
                                                      "nearest_real_R": g(0.01, 0.005, 0.02)},
                            "manifold_minus_linear_raw": {"probe_err_to_target": g(raw_gap, raw_gap - 0.01,
                                                                                   raw_gap + 0.01)}}})


def test_aggregate_blocks():
    docs = [fake(1, -0.02, (-0.03, -0.01), 0.04), fake(2, 0.01, (-0.01, 0.02), 0.06), fake(1, 0.04, (0.01, 0.05), 0.05)]
    out = sps.aggregate_blocks(docs)
    e = out["pairs"]["spline_minus_chord"]["probe_err_to_target"]
    assert e["mean"] == pytest.approx(0.01) and e["sd"] == pytest.approx(0.03)
    assert e["n_spline_ahead"] == 1                      # lower error is better: only -0.02
    assert (e["n_ci_excl_0"], e["n_ci_above_0"], e["n_ci_below_0"]) == (2, 1, 1)
    assert e["mean_over_probe_test_error"] == pytest.approx(0.1)
    r = out["pairs"]["spline_minus_chord"]["nearest_real_R"]
    assert r["n_spline_ahead"] == 3 and r["n_ci_above_0"] == 3   # higher R is better
    raw = out["pairs"]["spline_minus_raw_chord"]["probe_err_to_target"]
    assert raw["mean"] == pytest.approx(0.05) and raw["n_spline_ahead"] == 0 and raw["n_ci_excl_0"] == 3
    assert "nearest_real_R" not in out["pairs"]["spline_minus_raw_chord"]
    assert out["n_blocks"] == 3 and out["n_distinct_blocks"] == 2

"""scripts/run_objective_axis.py on a tmp results dir with some pieces missing (no VideoMAE, no step 2/3 for random)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fake_sweep(peak_score, peak, kind="circular"):
    cv = [peak_score * min(1.0, 0.8 + 0.2 * p / max(peak, 1)) for p in range(26)]
    layers = [{"point": p, "cv_mean": cv[p], "test_r2": cv[p] - 0.01, "cv_mae_mean": 5.0} for p in range(26)]
    return {"kind": kind, "layers": layers, "availability": {"peak": peak, "onset": 2, "selectivity_onset": 3}}


def test_objective_axis_missing_pieces(tmp_path):
    res, figs = tmp_path / "results", tmp_path / "figures"
    res.mkdir()
    (res / "p1a_direction_direction_meanpool.json").write_text(json.dumps(fake_sweep(0.99, 20)))
    (res / "p1a_direction_direction_meanpool_random.json").write_text(json.dumps(fake_sweep(0.85, 10)))
    (res / "p1b_direction_direction_meanpool_L20.json").write_text(json.dumps({"K": 50, "dims": 100}))
    single = [{"n": n, "mae_to_target": 90.0 / (1 + n)} for n in range(51)]
    (res / "p1c_direction_L20.json").write_text(json.dumps({"K": 50, "single": single}))
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_objective_axis.py"), "--results", str(res),
                           "--figures", str(figs), "--variables", "direction/direction"],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    assert "missing" in proc.stdout and "videomae" in proc.stdout
    out = json.loads((res / "objective_axis.json").read_text())
    rows = out["variables"]["direction/direction"]
    assert rows["videomae"] == {"available": False}
    v = rows["vjepa2"]
    assert v["peak_point"] == 20 and v["nested_K"] == 50 and v["steer_probes_to_bar"] == 8   # 90/(1+n) <= 10 at n=8
    assert abs(v["selectivity_at_peak"] - (0.99 - 0.85)) < 1e-9
    assert rows["random"]["nested_K"] is None and rows["random"]["steer_probes_to_bar"] is None
    assert (figs / "fig5_objective_axis.png").exists()

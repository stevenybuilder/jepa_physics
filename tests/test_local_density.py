"""scripts/run_local_density.py: Levina-Bickel on planted manifolds and midpoint density on a planted hollow ring."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_local_density", ROOT / "scripts" / "run_local_density.py")
ld = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ld)


def embed(P, D=64, noise=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((D, P.shape[1])))
    return P @ Q.T + noise * rng.standard_normal((len(P), D))


def test_lb_planted_2d_ring_is_2d():
    rng = np.random.default_rng(0)
    th, r = rng.uniform(0, 2 * np.pi, 2000), rng.uniform(0.7, 1.3, 2000)
    X = embed(np.stack([r * np.cos(th), r * np.sin(th)], 1))
    dim = ld.lb_dimension(ld.lb_inverse(X, 10))
    assert 1.7 < dim < 2.3, dim


def test_lb_thin_circle_is_1d_and_blob_is_higher():
    rng = np.random.default_rng(1)
    th = rng.uniform(0, 2 * np.pi, 1500)
    # noise well below the neighbour spacing (2 pi / 1500); LB reads the noise dimension once noise dominates
    circle = embed(np.stack([np.cos(th), np.sin(th)], 1), noise=1e-5)
    assert 0.8 < ld.lb_dimension(ld.lb_inverse(circle, 10)) < 1.25
    assert ld.lb_dimension(ld.lb_inverse(embed(rng.standard_normal((1500, 6))), 10)) > 4.5


def test_density_contrast_orders_hollow_ring():
    """Real points on a hollow ring; the chord midpoint (centre) is sparser than the ring midpoint."""
    rng = np.random.default_rng(2)
    th = rng.uniform(0, 2 * np.pi, 1500)
    r = rng.normal(1.0, 0.08, 1500)
    ref = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    q_real = ref[:50] + rng.normal(0, 0.01, (50, 2))
    q_spline = np.stack([np.cos(th[:50] + 0.3), np.sin(th[:50] + 0.3)], 1)
    q_chord = rng.normal(0, 0.05, (50, 2))
    d = {k: ld.knn_mean_dist(q, ref[50:], 5) for k, q in (("c", q_chord), ("s", q_spline), ("r", q_real))}
    out = ld.density_contrast(d["c"], d["s"], d["r"], np.arange(50), np.percentile(d["r"], 95))
    assert out["chord_over_real_geomean"] > 5 * out["spline_over_real_geomean"]
    assert 0.5 < out["spline_over_real_geomean"] < 2.0
    assert out["chord_minus_spline_ci95"][0] > 0
    assert out["frac_chord_beyond_real_p95"] == 1.0


def fake_direction(tmp, D=24, per=10, radius=5.0, noise=0.3):
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(64) * 360.0 / 64, per)
    X = rng.standard_normal((len(y), 3, D)).astype(np.float32) * noise
    X[:, 1, 0] += radius * np.cos(np.radians(y))
    X[:, 1, 1] += radius * np.sin(np.radians(y))
    ids = np.arange(len(y))
    (tmp / "act").mkdir()
    np.save(tmp / "act" / "meanpool.npy", X)
    (tmp / "act" / "ids.json").write_text(json.dumps(ids.tolist()))
    pd.DataFrame({"id": ids, "label": y, "theta_degrees": y, "speed_mps": 1.0, "acceleration_mps2": 0.0,
                  "motion": "velocity"}).to_csv(tmp / "table.csv", index=False)
    fold = {}
    for v in np.unique(y):
        for j, i in enumerate(ids[y == v]):
            fold[str(i)] = -1 if j < 2 else (j % 5)
    (tmp / "split.json").write_text(json.dumps({"fold": fold}))
    return ["--act-dir", str(tmp / "act"), "--table", str(tmp / "table.csv"), "--split", str(tmp / "split.json")]


def test_script_end_to_end_hollow_ring(tmp_path):
    io = fake_direction(tmp_path)
    out = ld.run(ld.parse(["--layers", "1", "--k", "16", *io, "--out", str(tmp_path / "r.json"),
                           "--figure", str(tmp_path / "f.png")]))
    e = out["layers"]["1"]
    assert (tmp_path / "f.png").exists()
    for sp in ("pca64", "leace2w", "chart2"):
        t = e["density"]["train"][sp]["180"]
        assert t["chord_minus_spline_logratio"] > 0 and t["chord_minus_spline_ci95"][0] > 0, sp
    assert e["intrinsic_dimension"]["chart2"]["all"]["k10"]["id"] < 2.3

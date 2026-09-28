"""scripts/run_accel_grid.py: the decorrelated design and the partialled / nested readouts (pure helpers)."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("accel_grid", ROOT / "scripts" / "run_accel_grid.py")
ag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ag)


def test_design_decorrelates_a_and_keeps_disk_on_screen():
    plan = ag.design(per_cell=4, seed=0)
    a = np.array([c["acceleration_mps2"] for c in plan])
    m = np.array([c["mean_speed_mps"] for c in plan])
    assert len(plan) == 4 * len(ag.GRID_M) * len(ag.GRID_A)
    assert abs(np.corrcoef(a, m)[0, 1]) < 1e-9
    for c in plan:
        xy = ag.trajectory(c["start_position_xy_m"], c["theta_degrees"], c["speed_mps"], c["acceleration_mps2"])
        assert np.abs(xy).max() <= ag.LIM
        s = np.linalg.norm(xy[-1] - xy[0])
        assert abs(s - c["displacement_m"]) < 1e-9                   # displacement = m x 0.625 s
        assert c["speed_mps"] > 0 and c["final_speed_mps"] > 0
    from wm.render_twin import trajectory_m
    c = plan[7]
    assert np.allclose(trajectory_m(c), ag.trajectory(c["start_position_xy_m"], c["theta_degrees"], c["speed_mps"],
                                                        c["acceleration_mps2"]))


def test_folds_balance_cells():
    plan = ag.design(per_cell=10, seed=0)
    f = ag.folds_by_cell([c["cell"] for c in plan])
    for cell in {tuple(c["cell"]) for c in plan}:
        idx = [i for i, c in enumerate(plan) if tuple(c["cell"]) == cell]
        assert sorted(np.bincount(f[idx], minlength=5)) == [2] * 5


def test_partial_ridge_sees_only_what_is_beyond_the_nuisance():
    rng = np.random.default_rng(0)
    n = 300
    m, a = rng.uniform(1, 3, n), rng.uniform(-3, 3, n)
    folds = np.arange(n) % 5
    W = rng.standard_normal((2, 40))
    X_speed_only = np.column_stack([m, m ** 2]) @ W + 0.05 * rng.standard_normal((n, 40))
    X_both = np.column_stack([m, a]) @ W + 0.05 * rng.standard_normal((n, 40))
    y, p = ag.partial_ridge_oof(X_speed_only, a, m[:, None], folds)
    assert ag.r2(y, p) < 0.1
    y, p = ag.partial_ridge_oof(X_both, a, m[:, None], folds)
    assert ag.r2(y, p) > 0.9


def test_nested_step_decode_is_out_of_fold():
    rng = np.random.default_rng(1)
    n, D = 60, 12
    S = rng.uniform(0, 4, (n, 8))
    w = rng.standard_normal(D)
    T = S[..., None] * w + 0.01 * rng.standard_normal((n, 8, D))
    folds = np.arange(n) % 5
    test, train = ag.nested_step_decode(T, S, folds)
    assert ag.r2(S.ravel(), test.ravel()) > 0.99
    for f, tr in train.items():
        assert tr.shape == ((folds != f).sum(), 8)

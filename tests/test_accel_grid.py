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


def test_magnitude_scores_order_blind_code_hides_sign_but_not_magnitude():
    """QA #358: features symmetric under time reversal (they see |a| and mean speed, not the sign of a) give ~0 R^2
    for signed a but recover |a|; features that see the sign recover both."""
    rng = np.random.default_rng(0)
    per = 12
    m = np.repeat([1.0, 1.75, 2.5, 3.25], 5 * per)
    a = np.tile(np.repeat([-3.0, -1.5, 0.0, 1.5, 3.0], per), 4)
    disp = m * ag.T_CLIP
    cells = [(i, j) for i in range(4) for j in range(5) for _ in range(per)]
    folds = ag.folds_by_cell(cells)
    nuis = np.column_stack([m, disp])
    W = rng.standard_normal((2, 30))
    X_blind = np.column_stack([np.abs(a), m]) @ W + 0.1 * rng.standard_normal((len(a), 30))
    r = ag.magnitude_scores(X_blind, a, nuis, folds, nb=200)
    assert r["signed_a"]["ridge_partial_mean_speed_displacement"]["r2"] < 0.05
    assert r["abs_a"]["ridge_partial_mean_speed_displacement"]["r2"] > 0.9
    assert r["abs_minus_signed_partial"]["ci95"][0] > 0.8
    X_signed = np.column_stack([a, m]) @ W + 0.1 * rng.standard_normal((len(a), 30))
    r = ag.magnitude_scores(X_signed, a, nuis, folds, nb=200)
    assert r["signed_a"]["ridge_partial_mean_speed_displacement"]["r2"] > 0.9
    # a linear readout of a signed code cannot fold it into |a|: |a| is uncorrelated with a on this grid
    assert r["abs_a"]["ridge_partial_mean_speed_displacement"]["r2"] < 0.05


def test_abs_of_signed_recovers_magnitude_from_a_signed_code():
    rng = np.random.default_rng(2)
    n = 200
    a = np.repeat([-3.0, -1.5, 0.0, 1.5, 3.0], n // 5)
    folds = np.arange(n) % 5
    X = a[:, None] * rng.standard_normal(20) + 0.1 * rng.standard_normal((n, 20))
    p = ag.abs_of_signed_oof(X, a, folds)
    assert ag.r2(np.abs(a), p) > 0.9


def test_loco_folds_never_share_a_cell_between_train_and_test():
    plan = ag.design(per_cell=6, seed=0)
    cells = [tuple(c["cell"]) for c in plan]
    f = ag.loco_folds(cells)
    assert f.max() + 1 == len(ag.GRID_M) * len(ag.GRID_A)
    for k in np.unique(f):
        te = {c for c, g in zip(cells, f) if g == k}
        tr = {c for c, g in zip(cells, f) if g != k}
        assert len(te) == 1 and not te & tr


def test_cartesian_targets_and_partial_scores():
    a, th = np.array([2.0, -1.0]), np.array([90.0, 0.0])
    ax, ay = ag.cartesian_targets(a, th)
    assert np.allclose(ax, [0, -1]) and np.allclose(ay, [2, 0])
    rng = np.random.default_rng(3)
    plan = ag.design(per_cell=12, seed=0)
    a = np.array([c["acceleration_mps2"] for c in plan])
    m = np.array([c["mean_speed_mps"] for c in plan])
    th = np.array([c["theta_degrees"] for c in plan])
    ax, ay = ag.cartesian_targets(a, th)
    X = np.column_stack([ax, ay, m]) @ rng.standard_normal((3, 30)) + 0.05 * rng.standard_normal((len(a), 30))
    r = ag.partial_targets_scores(X, a, th, np.column_stack([m, m * ag.T_CLIP]),
                                  ag.loco_folds([c["cell"] for c in plan]), nb=100)
    assert r["cartesian"]["mean"]["r2"] > 0.9 and r["signed_a"]["r2"] < 0.2


def test_cellblock_ci_is_wider_when_clips_in_a_cell_share_error():
    rng = np.random.default_rng(4)
    cells = [(i, 0) for i in range(20) for _ in range(12)]
    y = rng.standard_normal(240)
    p = y + np.repeat(rng.standard_normal(20), 12)                  # error shared within each cell
    cb = ag.cellblock_ci([y], [p], cells, nb=300)["components"][0]
    clip = ag.boot_r2(y, p, nb=300)["ci95"]
    assert cb[1] - cb[0] > clip[1] - clip[0]

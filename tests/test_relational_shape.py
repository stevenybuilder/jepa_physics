import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_relational_shape as rs  # noqa: E402


def test_straight_line_has_zero_residual_and_ordered():
    rng = np.random.default_rng(0)
    b = rng.standard_normal(50)
    levels = list(range(7))
    C = np.stack([2.0 + l * b for l in levels])
    m, slope, _ = rs.line_metrics(C, levels)
    assert m["resid_frac"] < 1e-12 and m["spearman_pc1"] == 1.0 and m["n_inversions"] == 0
    assert np.allclose(slope, b)


def test_curved_manifold_has_residual():
    levels = list(range(7))
    C = np.stack([[l, (l - 3) ** 2, 0.0] for l in levels], dtype=float)
    assert rs.line_metrics(C, levels)[0]["resid_frac"] > 0.3


def test_principal_angle():
    B = np.eye(4)[:, :2]
    assert abs(rs.principal_angle(np.array([1.0, 1, 0, 0]), B)) < 1e-6
    assert abs(rs.principal_angle(np.array([0.0, 0, 1, 0]), B) - 90) < 1e-6


def test_ridge_recovers_linear_target():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((120, 30))
    w = rng.standard_normal(30)
    y = X @ w + 0.5
    folds = np.arange(120) % 5
    al = rs.ridge_cv_alpha(X, y, folds)
    wh, bh = rs.ridge_fit(X, y, al)
    assert rs.r2(y, X @ wh + bh) > 0.99

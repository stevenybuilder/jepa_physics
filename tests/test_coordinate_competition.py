import importlib.util
from pathlib import Path

import numpy as np

_s = importlib.util.spec_from_file_location("cc", Path(__file__).resolve().parents[1] / "scripts" / "run_coordinate_competition.py")
cc = importlib.util.module_from_spec(_s)
_s.loader.exec_module(cc)


def test_ranks_match_columns():
    th, v = np.linspace(0, 350, 36), np.linspace(1, 4, 36)
    for n, k in cc.CANDIDATES.items():
        assert cc.phi(n, th, v).shape == (36, k), n


def test_true_frame_wins_rank_matched():
    rng = np.random.default_rng(0)
    th, v = rng.uniform(0, 360, 800), rng.uniform(0.5, 4, 800)
    M = rng.standard_normal((2, 64))
    Z = cc.phi("cartesian", th, v) @ M + 0.3 * rng.standard_normal((800, 64))
    k, p = slice(0, 500), slice(500, 800)
    mu, V = cc._pca(Z[k])
    Z = (Z - mu) @ V                              # score_block expects knot-PCA coordinates
    bi = rng.integers(0, 300, (50, 300))
    o, b = cc.score_block(Z[k], th[k], v[k], Z[p], th[p], v[p], bi, rng)
    s = cc.summarise(o, b)
    assert s["winner_rank2"] == "cartesian"
    assert s["h2h_rank2_polar2_minus_cartesian"]["diff"] < 0
    assert o["pca_2"] >= o["cartesian"] - 0.02    # rank-k PCA is the ceiling

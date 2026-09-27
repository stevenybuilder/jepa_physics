"""Step 1 support runs (spec 5.1 a-c) on planted codes."""
from functools import partial

import numpy as np

from wm.probes import score
from wm.support import onset_compare, spatial_curve, transfer_curve

N_LAYERS = 25


def sigmoid(p, mid, width=1.0):
    return 1 / (1 + np.exp(-(p - mid) / width))


def planted_layers(code_fns, n, d=24, seed=0, noise=0.3):
    """X_by_layer(p) [n, d]: sum over (code [n, k], mid) of code·sigmoid(p − mid) in its own random
    slots, plus isotropic noise; one fixed rotation."""
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.standard_normal((d, d)))[0]
    eps = [rng.standard_normal((n, d)) * noise for _ in range(N_LAYERS + 1)]
    def get_X(p):
        cols = [code * sigmoid(p, mid) for code, mid in code_fns]
        Z = np.hstack(cols + [np.zeros((n, d - sum(c.shape[1] for c in cols)))])
        return Z @ basis.T + eps[p]
    return get_X


def test_cartesian_before_polar_on_planted_onsets():
    rng = np.random.default_rng(0)
    n = 900
    th, sp = rng.uniform(0, 2 * np.pi, n), rng.uniform(1, 7, n)
    vxvy = sp[:, None] * np.stack([np.cos(th), np.sin(th)], 1)
    sincos = np.stack([np.sin(th), np.cos(th)], 1)
    get_X = planted_layers([(vxvy / 4, 3), (sincos, 14)], n)          # (vx, vy) from ~3, (sin, cos) from ~14
    rows, folds = np.arange(n), np.arange(n) % 5
    res = onset_compare(get_X, {"vxvy": (vxvy, partial(score, kind="vector")),
                                "sincos": (sincos, partial(score, kind="circular"))}, rows, folds, n_boot=50)
    a, b = res["targets"]["vxvy"]["availability"], res["targets"]["sincos"]["availability"]
    assert a["onset"] < b["onset"] and b["onset"] >= 10
    assert a["onset_ci"][0] <= a["onset"] <= a["onset_ci"][1]
    diff = res["onset_difference"]
    assert diff["point_estimate"] < 0 and diff["ci95"][1] < 0 and diff["frac_draws_first_earlier"] == 1.0


def test_transfer_high_for_shared_code_chance_for_foreign_code():
    rng = np.random.default_rng(1)
    th = [rng.uniform(0, 2 * np.pi, m) for m in (800, 400, 400)]
    Y = [np.stack([np.sin(t), np.cos(t)], 1) for t in th]
    src = planted_layers([(Y[0], 5)], 800, seed=3, noise=0.1)
    same = planted_layers([(Y[1], 5)], 400, seed=3, noise=0.1)                   # same rotation = same code
    foreign = planted_layers([(np.hstack([0 * Y[2], Y[2]]), 5)], 400, seed=3, noise=0.1)   # orthogonal slots
    sub = {"first_half": np.arange(400) < 200}
    rows = transfer_curve(src, Y[0], np.arange(800), np.arange(800) % 5,
                          {"same": (same, Y[1], sub), "foreign": (foreign, Y[2], None)}, points=[0, 12])
    late = rows[1]
    assert late["source_cv_r2"] > 0.9 and late["targets"]["same"]["r2"] > 0.85
    assert late["targets"]["same"]["mae"] < 10 and late["targets"]["same"]["first_half"]["n"] == 200
    assert late["targets"]["foreign"]["mae"] > 60
    assert rows[0]["targets"]["same"]["r2"] < late["targets"]["same"]["r2"]


def test_spatial_generalisation_fails_only_for_position_specific_code():
    rng = np.random.default_rng(2)
    n = 800
    th = rng.uniform(0, 2 * np.pi, n)
    Y = np.stack([np.sin(th), np.cos(th)], 1)
    x = rng.uniform(-2, 2, n)
    neg, pos = np.flatnonzero(x < 0), np.flatnonzero(x > 0)
    shared = planted_layers([(Y, 2)], n, seed=5, noise=0.1)
    # Position-specific: x < 0 clips carry the code in slots 0-1, x > 0 clips in slots 2-3.
    split_code = np.hstack([Y * (x < 0)[:, None], Y * (x > 0)[:, None]])
    specific = planted_layers([(split_code, 2)], n, seed=5, noise=0.1)
    f_neg, f_pos = np.arange(len(neg)) % 5, np.arange(len(pos)) % 5
    s = spatial_curve(shared, Y, neg, pos, f_neg, f_pos, points=[12])[0]
    p = spatial_curve(specific, Y, neg, pos, f_neg, f_pos, points=[12])[0]
    for key in ("neg_to_pos", "pos_to_neg"):
        assert s[key]["within_cv_r2"] > 0.9 and s[key]["cross_r2"] > 0.85
        assert p[key]["within_cv_r2"] > 0.9 and p[key]["cross_mae"] > 60


def test_support_script_and_figures_on_fake_activations(tmp_path):
    import json

    from fake_acts import write_fake_meanpool
    from test_end_to_end import run
    acts, results, figs = tmp_path / "acts", tmp_path / "results", tmp_path / "figs"
    for dataset in ("direction", "speed", "acceleration"):
        write_fake_meanpool(acts, dataset)
    run("run_step1_support.py", "--pools", "meanpool", "--n-boot", "20", "--act-root", str(acts),
        "--results", str(results))
    onset = json.loads((results / "p1a_support_onset_direction_meanpool.json").read_text())
    assert onset["n_clips"] < 750 and set(onset["targets"]) == {"vxvy", "sincos"}
    tr = json.loads((results / "p1a_support_transfer_meanpool.json").read_text())
    assert set(tr["layers"][0]["targets"]) == {"speed", "acceleration"} and len(tr["layers"]) == 26
    assert "speed_below_1" in tr["layers"][12]["targets"]["speed"]
    sp = json.loads((results / "p1a_support_spatial_meanpool.json").read_text())
    assert sp["n_neg"] + sp["n_pos"] <= 1200 and sp["layers"][12]["neg_to_pos"]["within_cv_r2"] > 0.5
    run("make_figures.py", "--results", str(results), "--figures", str(figs))
    for name in ("fig1c_direction_transfer", "fig1d_cartesian_vs_polar", "fig1e_spatial_generalisation"):
        assert (figs / f"{name}.png").exists(), name

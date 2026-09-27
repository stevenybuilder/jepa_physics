"""Four dimensionality estimands (spec §6 item 8) on planted rings."""
from functools import partial

import numpy as np
import pytest

from wm.dims import leace_fit, leace_scores, planted_ring, split_half_spectrum, whitened_count
from wm.inlp import inlp
from wm.probes import ALPHAS, cv_select_alpha, score

FOLDS = np.arange(1200) % 5


def sincos(theta):
    r = np.radians(theta)
    return np.stack([np.sin(r), np.cos(r)], 1)


def literal_K(X, Y):
    sf = partial(score, kind="circular")
    alpha = cv_select_alpha(X, Y, FOLDS, ALPHAS, sf)["alpha"]
    return inlp(X, Y, X[:5], Y[:5], FOLDS, alpha, sf, "circular")[0]["K"]


@pytest.fixture(scope="module")
def rings():
    return {"clean": planted_ring(), "sheared": planted_ring(shear=30)}


def test_whitened_count_is_2_and_literal_count_inflates_under_shear(rings):
    Xc, thc = rings["clean"]
    Xs, ths = rings["sheared"]
    assert whitened_count(Xc, sincos(thc), FOLDS, "circular")["K"] == 1
    assert whitened_count(Xs, sincos(ths), FOLDS, "circular")["K"] == 1
    k_clean, k_shear = literal_K(Xc, sincos(thc)), literal_K(Xs, sincos(ths))
    assert k_clean == 1 and k_shear >= 3


def test_leace_erases_linear_information(rings):
    X, th = rings["sheared"]
    Y = sincos(th)
    Xe = leace_fit(X, Y)(X)
    cov = (Xe - Xe.mean(0)).T @ (Y - Y.mean(0)) / len(X)
    assert np.abs(cov).max() < 1e-8                          # no cross-covariance left (LEACE guarantee)
    s = leace_scores(X, Y, FOLDS, "circular", mlp=False)
    assert s["ridge_r2_before"] > 0.8 and s["ridge_r2_after"] < 0.05


def test_split_half_spectrum_finds_planted_harmonics():
    X, th = planted_ring(harmonics={1: 1.0, 3: 0.7}, seed=1)
    s = split_half_spectrum(X, th)
    frac = np.array(s["frac_k_ge_1"])
    assert frac[0] > 0.5 and frac[2] > 0.2 and frac[[1, 3, 4, 5, 6, 7]].max() < 0.05
    assert 1.5 < s["participation_ratio"] < 2.2
    X1, th1 = planted_ring(seed=1)
    assert split_half_spectrum(X1, th1)["participation_ratio"] < 1.1


def test_dims_script_on_fake_activations_and_figure(tmp_path):
    import json

    from fake_acts import write_fake_meanpool
    from test_end_to_end import run
    acts, results, figs = tmp_path / "acts", tmp_path / "results", tmp_path / "figs"
    for dataset in ("direction", "speed"):
        write_fake_meanpool(acts, dataset)
    run("run_step2_dims.py", "--points", "3", "12", "--no-mlp", "--no-control", "--seeds", "2", "--eps", "1e-2",
        "--act-root", str(acts), "--results", str(results))
    out = json.loads((results / "p1b_dims_four_ways.json").read_text())
    row = out["variables"]["direction"]["layers"][1]
    assert row["point"] == 12 and row["leace"]["ridge_r2_after"] < 0.1 and "dft" in row
    assert row["whitened"][0]["K"] >= 1
    assert "dft" not in out["variables"]["speed"]["layers"][0]
    run("make_figures.py", "--results", str(results), "--figures", str(figs))
    assert (figs / "fig2c_dims_four_ways.png").exists()


def test_mlp_recovers_nonlinear_target():
    from wm.dims import mlp_cv_r2
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 4))
    y = np.cos(2 * X[:, 0]) + 0.05 * rng.standard_normal(400)     # no linear information
    assert mlp_cv_r2(X, y, np.arange(400) % 5, n_folds=1, max_iter=1000) > 0.5


def test_whitened_count_nested_folds_paper_variant_and_no_fold_leak(rings):
    X, th = rings["sheared"]
    Y = sincos(th)
    tr = np.arange(1200) < 1000
    out = whitened_count(X[tr], Y[tr], FOLDS[tr], "circular", Xte=X[~tr], Yte=Y[~tr])
    assert out["K"] == 1 and out["protocol"] == "nested" and out["K_folds"] == [1] * 5 and out["paper"]["K"] == 1
    assert len(out["cv_r2_folds_by_round"][0]) == 5 and len(out["alpha_folds"]) == 5
    # permuting the held-out fold's labels in the fit labels leaves that fold's curve unchanged
    val = FOLDS[tr] == 0
    Yp = Y[tr].copy()
    Yp[val] = Yp[val][np.random.default_rng(3).permutation(val.sum())]
    perm = whitened_count(X[tr], Y[tr], FOLDS[tr], "circular", Yfit=Yp)
    n = min(len(out["cv_r2_folds_by_round"]), len(perm["cv_r2_folds_by_round"]))
    assert np.allclose([r[0] for r in out["cv_r2_folds_by_round"][:n]],
                       [r[0] for r in perm["cv_r2_folds_by_round"][:n]], atol=1e-10)


def test_bakeoff_refit_inlp_basis_reads_no_test_rows(monkeypatch):
    import wm.inlp as inlp_mod
    from wm.bakeoff import refit_inlp_basis
    import pandas as pd
    X, th = planted_ring(seed=2)
    n = len(X)
    d = {"X": X, "df": pd.DataFrame({"theta_degrees": th}), "fold": np.arange(n) % 5,
         "role": np.where(np.arange(n) < n - 200, "train", "test")}
    seen = {}
    real = inlp_mod.inlp

    def spy(Xtr, Ytr, Xte, Yte, *a, **k):
        seen["Xte"] = Xte
        return real(Xtr, Ytr, Xte, Yte, *a, **k)
    monkeypatch.setattr(inlp_mod, "inlp", spy)
    basis, _, note = refit_inlp_basis(d, d["role"] == "train", "direction")
    assert seen["Xte"] is None and note["n_probes"] >= 1 and basis["Q"].shape[1] == 2 * note["n_probes"]

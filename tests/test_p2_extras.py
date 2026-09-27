"""Small tests of the three Part 2 extras: donor-transplant ceiling, two-route 180-degree test, rotating-code detector."""
import importlib.util
import json

import numpy as np

from test_p2_scripts import ROOT, fake_dataset


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def argv(tmp):
    return ["--layers", "1", "--k", "16", "--act-dir", str(tmp / "act"), "--table", str(tmp / "table.csv"),
            "--split", str(tmp / "split.json"), "--results-dir", str(tmp / "results"),
            "--figures-dir", str(tmp / "figures")]


def test_transplant_decomposition():
    from wm.manifold import fit_pca
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 20))
    pca = fit_pca(X, 5)
    mod = load("run_donor_ceiling")
    in_sub, full = mod.transplants(X[:3], X[3], pca)
    np.testing.assert_allclose(pca.project(in_sub), np.broadcast_to(pca.project(X[3]), (3, 5)), atol=1e-8)
    np.testing.assert_allclose(pca.complement(in_sub), pca.complement(X[:3]), atol=1e-8)
    np.testing.assert_allclose(full, np.broadcast_to(X[3], (3, 20)))


def test_donor_ceiling_end_to_end(tmp_path):
    fake_dataset(tmp_path, "direction")
    mod = load("run_donor_ceiling")
    out = mod.run(mod.parse(argv(tmp_path)))
    s = out["layers"]["1"]["summary"]
    assert set(s) == {"donor_in_subspace", "donor_full", "spline", "chord"}
    assert s["donor_full"]["probe_err"] < 45 and "verdict" in out
    assert (tmp_path / "results" / "p2_donor_ceiling_direction_L1_contiguous.json").exists()


def test_two_route_separates_on_planted_ring(tmp_path):
    fake_dataset(tmp_path, "direction")
    mod = load("run_two_route")
    out = mod.run(mod.parse(argv(tmp_path)))
    r = out["layers"]["1"]["routes"]
    assert r["via_plus90"]["midpoint"]["probe_frac_closer_to_plus90"] > 0.8
    assert r["via_minus90"]["midpoint"]["probe_frac_closer_to_plus90"] < 0.2
    assert r["chord"]["midpoint"]["probe_radius_mid_mean"] < 0.5 * r["chord"]["midpoint"]["probe_radius_start_mean"]
    assert json.loads((tmp_path / "results" / "p2_two_route_direction_L1.json").read_text())["verdict"]


def test_route_steps_go_opposite_ways():
    mod = load("run_two_route")

    class C:   # identity-coordinate ring: intrinsic angle = label angle in radians
        values = np.arange(64) * 360.0 / 64

        @staticmethod
        def coord_of_value(v):
            return np.radians(np.asarray(v, float)) % (2 * np.pi)
    _, plus, minus = mod.route_steps(C, np.array([10.0, 200.0]))
    np.testing.assert_allclose(plus, np.pi, atol=1e-9)
    np.testing.assert_allclose(minus, -np.pi, atol=1e-9)


def test_rotating_detector_flags_rotating_not_constant():
    mod = load("run_rotating_speed_axis")
    rng = np.random.default_rng(0)
    n, D = 1280, 40
    X = rng.standard_normal((n, D))
    theta = rng.uniform(0, 360, n)
    speed = rng.uniform(0.5, 4.0, n)
    fold = np.arange(n) % 5
    Xr, _ = mod.plant(X, theta, speed, "rotating", 1.0, ref_sd=1.0)
    Xc, _ = mod.plant(X, theta, speed, "constant", 1.0, ref_sd=1.0)
    assert mod.detect(Xr, speed, theta, fold, k=16)["rotates"]
    assert not mod.detect(Xc, speed, theta, fold, k=16)["rotates"]

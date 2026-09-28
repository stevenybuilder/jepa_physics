"""End-to-end run of scripts/run_geometry_checks.py and scripts/run_part2.py on a tiny fake meanpool."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
N_LAYERS, D, PER_VALUE = 3, 32, 8


def fake_dataset(tmp, dataset):
    """64 values x 8 clips; layer 1 holds a planted ring (direction) or line (speed); a split with 5 train folds."""
    rng = np.random.default_rng(0)
    if dataset == "direction":
        y = np.repeat(np.arange(64) * 360.0 / 64, PER_VALUE)
        r = np.radians(y)
        signal = np.zeros((len(y), D))
        signal[:, 0], signal[:, 1] = 5 * np.cos(r), 5 * np.sin(r)
        table = {"theta_degrees": y, "speed_mps": 1.0, "acceleration_mps2": 0.0,
                 "motion": np.where(np.arange(len(y)) % 2, "velocity", "acceleration")}
    else:
        y = np.repeat(np.linspace(0.25, 4.0, 64), PER_VALUE)
        signal = np.zeros((len(y), D))
        signal[:, 0] = 3 * y
        table = {"theta_degrees": 0.0, "speed_mps": y, "acceleration_mps2": 0.0, "motion": "velocity"}
    n = len(y)
    ids = np.arange(n)
    X = rng.standard_normal((n, N_LAYERS, D)).astype(np.float32) * 0.3
    X[:, 1] += signal
    act = tmp / "act"
    act.mkdir()
    np.save(act / "meanpool.npy", X)
    (act / "ids.json").write_text(json.dumps(ids.tolist()))
    pd.DataFrame({"id": ids, "label": y, **table}).to_csv(tmp / "table.csv", index=False)
    # stratified: in each value, 2 of 8 clips are test, the other 6 cycle through folds 0..4
    fold = {}
    for v in np.unique(y):
        members = ids[y == v]
        for j, i in enumerate(members):
            fold[str(i)] = -1 if j < 2 else (j + int(v * 10)) % 5
    (tmp / "split.json").write_text(json.dumps({"fold": fold}))
    return tmp


def run_script(name, tmp, dataset, *extra):
    """Run a script's parse + run in-process with command-line arguments (same code path as the CLI)."""
    spec = importlib.util.spec_from_file_location(name[:-3], ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = ["--dataset", dataset, "--layer", "1", "--k", "16",
            "--act-dir", str(tmp / "act"), "--table", str(tmp / "table.csv"), "--split", str(tmp / "split.json"),
            "--results-dir", str(tmp / "results"), "--figures-dir", str(tmp / "figures"), *extra]
    module.run(module.parse(argv))


@pytest.mark.parametrize("dataset", ["direction", "speed"])
def test_scripts_end_to_end(tmp_path, dataset):
    tmp = fake_dataset(tmp_path, dataset)
    run_script("run_geometry_checks.py", tmp, dataset)
    geo = json.loads((tmp / "results" / f"p2_geometry_{dataset}_L1.json").read_text())
    assert (tmp / "figures" / f"fig4_centroid_plane_{dataset}_L1.png").exists()
    assert set(geo["loo"]) == {"1", "2", "4", "8", "16"}
    if dataset == "direction":
        assert geo["angle"]["source"].startswith("unsupervised")
        assert geo["loo"]["16"]["fp32"]["winner"] == "cubic"
    else:
        assert geo["knot_spacing"]["better"] == "linear"

    run_script("run_part2.py", tmp, dataset, "--variable", dataset, "--n-clips", "12", "--K", "11")
    tag = f"{dataset}_{dataset}_L1_scattered"
    res = json.loads((tmp / "results" / f"p2_steer_{tag}.json").read_text())
    for fig in ("gap_vs_shift", "path_energy"):
        assert (tmp / "figures" / f"fig4_{fig}_{tag}.png").exists()
    assert len(res["held_out_values"]) == 16 and res["holdout"]["design"] == "scattered"
    s = res["summary"]
    # the real steer beats every shuffled-centroid draw
    shuf = res["controls"]["shuffled_unmatched"]["nearest_real_R"]
    assert shuf["n_draws"] == 20 and shuf["spline_rank"] == 1
    assert s["manifold"]["overall"]["nearest_real_R"] > shuf["p95"]
    assert res["K"] == 11 and "behaviour_manifold" in res and "deviations_from_goodfire" in res
    assert res["summary_notes"][0].startswith("verdict: ") and res["energy_floor"]["to_curve"] > 0
    if dataset == "speed":
        assert res["verdict"]["call"] in ("negative", "no_curvature", "path_geometry_positive")   # no indep gain
        assert not res["verdict"]["independent_practical_gain_on"]
    else:
        assert "excess_to_curve" in res["verdict"]["spline_better_than_chord_on"]
    if dataset == "direction":
        far_lin = s["linear"]["by_shift"][-1]["excess_to_curve"]
        far_man = s["manifold"]["by_shift"][-1]["excess_to_curve"]
        assert far_lin > 1.0 and far_lin > 3 * abs(far_man)   # large shifts: the line leaves the ring
        wp = res["waypoint_readout"]
        lin, man = wp["linear"][-1], wp["manifold"][-1]      # the bin nearest 180 degrees
        assert len(lin["radius"]) == 11 and min(lin["radius"]) < 0.5 * min(man["radius"])   # radius collapse
        assert s["linear"]["by_shift"][-1]["behaviour_energy"] > 1.3 * s["manifold"]["by_shift"][-1]["behaviour_energy"]
        assert res["isometry_behaviour"] > 0.9 and res["isometry_probe"] > 0.9
        assert "manifold_transport" in s and "goodfire_manifold" in s     # Goodfire's replace arm runs by default
        g = res["gaps"]["manifold_minus_linear"]["excess_to_curve"]
        assert g["ci95"][1] < 0 and g["se_over_targets"] > 0 and len(g["by_shift"]) >= 3
        em = res["controls"]["random_endpoint_matched"]["excess_to_curve"]
        assert em["n_draws"] == 20
        assert [t["tau"] for t in res["tau_sensitivity"]] == [0.25, 0.5, 1.0, 2.0]
        assert res["behaviour_floor"]["mean"] > 0 and any("caveat" in n for n in res["summary_notes"])
        assert s["goodfire_manifold"]["overall"]["behaviour_energy"] is None
        assert res["gaps"]["manifold_minus_goodfire_manifold"]["behaviour_energy"].startswith("not comparable")
        vh = res["value_heatmap"]["arms"]
        assert np.asarray(vh["manifold"]["mass"]).shape == (11, 64)
        far = s["manifold"]["by_shift"][-1]["intermediate_mass"], s["linear"]["by_shift"][-1]["intermediate_mass"]
        assert far[0] > far[1]                                        # the spline sweeps, the line jumps
        assert (tmp / "figures" / f"fig4_value_heatmap_{tag}.png").exists()
        gf_rows = [r for r in res["rows"] if r["arm"] == "goodfire_manifold"]
        assert gf_rows and all(r[q] is None for r in gf_rows for q in
                               ("behaviour_energy", "behaviour_energy_rel_floor", "behaviour_entropy_mean",
                                "intermediate_mass"))
        assert all(r["nearest_real_R"] is not None for r in gf_rows)
        assert vh["goodfire_manifold"]["mass"] is None and vh["goodfire_manifold"]["note"].startswith("not comparable")
        assert res["waypoint_readout"]["goodfire_manifold"][-1]["intermediate_mass"].startswith("not comparable")
        op = s["manifold"]["over_pairs"]["behaviour_energy"]
        assert op["n_pairs"] > 16 and op["se"] > 0
        g = res["gaps"]["manifold_minus_linear"]["behaviour_energy"]
        assert g["n_pairs"] == op["n_pairs"] and g["se_over_pairs"] > 0
        far_o = s["manifold"]["by_shift"][-1]["ordering_spearman"], s["linear"]["by_shift"][-1]["ordering_spearman"]
        assert far_o[0] > 0.5 and far_o[0] > far_o[1]                 # the spline is ordered; the chord is not
        assert all(r["arc_sign"] in (1.0, -1.0) for r in res["rows"] if r["arm"] == "manifold")
        man_rows = [r for r in res["rows"] if r["arm"] == "manifold"]
        assert all(isinstance(r["behaviour_energy"], float) for r in man_rows)


@pytest.mark.parametrize("dataset,design", [("direction", "contiguous"), ("speed", "contiguous"),
                                            ("speed", "extrapolation")])
def test_part2_holdout_designs(tmp_path, dataset, design):
    tmp = fake_dataset(tmp_path, dataset)
    run_script("run_part2.py", tmp, dataset, "--variable", dataset, "--n-clips", "6", "--K", "5",
               "--holdout", design, "--n-controls", "3", "--goodfire-baseline", "--spline", "smooth", "--no-bf16")
    res = json.loads((tmp / "results" / f"p2_steer_{dataset}_{dataset}_L1_{design}.json").read_text())
    assert res["holdout"]["design"] == design and len(res["held_out_values"]) == 8
    assert res["spline"] == "smoothing"
    assert len(res["sagitta_per_target"]) == 8 and all(r["sagitta"] >= 0 for r in res["sagitta_per_target"])
    assert set(res["goodfire_comparison"]["summary"]) == {"goodfire_linear", "goodfire_manifold"}
    assert res["controls"]["random_endpoint_matched"]["nearest_real_R"]["n_draws"] == 3
    if design == "extrapolation":
        assert res["held_out_values"] == sorted(res["held_out_values"])[-8:]


def test_part2_heldout_context(tmp_path):
    """Direction spline from the direction set, steering clips of a second dataset with the same ring code."""
    tmp = fake_dataset(tmp_path, "direction")
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    fake_dataset(ctx, "direction")
    ids = json.loads((ctx / "act" / "ids.json").read_text())
    X = np.load(ctx / "act" / "meanpool.npy")
    X = X + 0.05 * np.random.default_rng(1).standard_normal(X.shape).astype(np.float32)
    np.save(ctx / "act" / "meanpool.npy", X)
    run_script("run_part2.py", tmp, "direction", "--variable", "direction", "--n-clips", "6", "--K", "5",
               "--n-controls", "2", "--no-bf16", "--context-dataset", "speed",
               "--context-act-dir", str(ctx / "act"), "--context-table", str(ctx / "table.csv"),
               "--context-split", str(ctx / "split.json"))
    res = json.loads((tmp / "results" / "p2_steer_direction_direction_L1_scattered_ctx-speed.json").read_text())
    assert res["context"]["steered_dataset"] == "speed" and len(ids) == len(X)
    man = res["summary"]["manifold"]["overall"]
    assert man["nearest_real_R_context"] > 0.3 and man["nearest_real_R"] > 0.3
    pd.read_csv(ctx / "table.csv").drop(columns=["motion"]).to_csv(ctx / "table.csv", index=False)
    run_script("run_part2.py", tmp, "direction", "--variable", "direction", "--n-clips", "6", "--K", "5",
               "--n-controls", "2", "--no-bf16", "--context-dataset", "speed", "--nuisance-regress",
               "--context-act-dir", str(ctx / "act"), "--context-table", str(ctx / "table.csv"),
               "--context-split", str(ctx / "split.json"))
    res = json.loads((tmp / "results" / "p2_steer_direction_direction_L1_scattered_ctx-speed_nuis.json").read_text())
    assert res["nuisance_regressed"]["context"]["covariates_filled_with_primary_train_mean"] == ["motion=velocity"]


def test_load_pair_checks_alignment(tmp_path):
    from wm.p2_data import load_pair
    tmp = fake_dataset(tmp_path, "direction")
    kw = dict(act_dirs=(tmp / "act", tmp / "act"), tables=(tmp / "table.csv", tmp / "table.csv"),
              splits=(tmp / "split.json", tmp / "split.json"))
    a, b = load_pair("direction", "speed", 1, "direction", **kw)
    assert a["X"].shape == b["X"].shape and b["periodic"]
    (tmp / "act" / "ids.json").write_text(json.dumps(list(range(1, 513))))
    with pytest.raises(AssertionError):
        load_pair("direction", "speed", 1, "direction", **kw)


def test_bakeoff_end_to_end(tmp_path):
    from sklearn.linear_model import Ridge
    from wm.inlp import save_basis
    tmp = fake_dataset(tmp_path, "direction")
    X = np.load(tmp / "act" / "meanpool.npy")[:, 1].astype(float)
    y = pd.read_csv(tmp / "table.csv")["theta_degrees"].to_numpy()
    fold = json.loads((tmp / "split.json").read_text())["fold"]
    tr = np.array([fold[str(i)] >= 0 for i in range(len(y))])
    Xs = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-8)
    Y = np.stack([np.sin(np.radians(y)), np.cos(np.radians(y))], 1)
    r = Ridge(alpha=1.0).fit(Xs[tr], Y[tr])
    W = r.coef_.T[None]                                            # one probe: [1, D, 2]
    save_basis(tmp / "basis.npz", np.linalg.qr(W[0])[0], W, r.intercept_[None], 1.0, 1, "circular")
    run_script("run_bakeoff.py", tmp, "direction", "--n-clips", "8", "--probe-basis", str(tmp / "basis.npz"))
    res = json.loads((tmp / "results" / "p2_bakeoff_direction_direction_L1_contiguous.json").read_text())
    assert "favours probe_qr" in res["probe_basis"]["rows"]
    run_script("run_bakeoff.py", tmp, "direction", "--n-clips", "8", "--max-probe-rounds", "3")
    res = json.loads((tmp / "results" / "p2_bakeoff_direction_direction_L1_contiguous.json").read_text())
    assert res["probe_basis"]["rows"] == "knot folds at kept values only" and res["probe_basis"]["folds"] == [0, 1, 2]
    assert res["provenance"]["holdout"] == "contiguous"
    arms = res["arms"]
    assert set(arms) == {"spline", "chord", "centroid_transport", "snap", "ring_rotation", "probe_qr"}
    assert arms["ring_rotation"]["nominal_rank"] == 2 and arms["ring_rotation"]["effective_rank"] < 2.5
    assert arms["probe_qr"]["nominal_rank"] % 2 == 0
    for e in ("probe", "mlp"):
        assert arms["spline"][f"err_{e}_matched"] < 0.5 * res["unsteered_err"][e]
    assert (tmp / "figures" / "fig4_bakeoff_rank_direction_direction_L1_contiguous.png").exists()


def velocity_dataset(tmp, plane):
    """Speed-set-like fake: 64 speeds x random directions; ring code (plane=False) or velocity code (plane=True)."""
    rng = np.random.default_rng(2)
    n = 64 * 16
    v = np.repeat(np.linspace(0.5, 4.0, 64), 16)
    th = rng.choice(np.arange(64) * 360.0 / 64, size=n)
    r = np.radians(th)
    amp = v if plane else np.full(n, 2.0)
    X = rng.standard_normal((n, N_LAYERS, D)).astype(np.float32) * 0.3
    X[:, 1, 0] += amp * np.cos(r)
    X[:, 1, 1] += amp * np.sin(r)
    X[:, 1, 2] += 0 if plane else 2.0 * v                          # ring: a separate speed axis
    act = tmp / "act"
    act.mkdir(parents=True)
    np.save(act / "meanpool.npy", X)
    ids = np.arange(n)
    (act / "ids.json").write_text(json.dumps(ids.tolist()))
    pd.DataFrame({"id": ids, "label": v, "theta_degrees": th, "speed_mps": v, "acceleration_mps2": 0.0,
                  "motion": "velocity"}).to_csv(tmp / "table.csv", index=False)
    fold = {str(i): (-1 if i % 5 == 0 else int(i % 5 - 1)) for i in ids}
    (tmp / "split.json").write_text(json.dumps({"fold": fold}))


@pytest.mark.parametrize("plane", [False, True])
def test_velocity_plane_script(tmp_path, plane):
    dir_tmp = fake_dataset(tmp_path, "direction")
    sp = tmp_path / "speed"
    sp.mkdir()
    velocity_dataset(sp, plane)
    spec = importlib.util.spec_from_file_location("vp", ROOT / "scripts" / "run_velocity_plane.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    res = module.run(module.parse([
        "--layers", "1", "--k", "16", "--n-clips", "60",
        "--dir-act-dir", str(dir_tmp / "act"), "--dir-table", str(dir_tmp / "table.csv"),
        "--dir-split", str(dir_tmp / "split.json"), "--speed-act-dir", str(sp / "act"),
        "--speed-table", str(sp / "table.csv"), "--speed-split", str(sp / "split.json"),
        "--results-dir", str(tmp_path / "results"), "--figures-dir", str(tmp_path / "figures")]))
    L = res["layers"]["1"]
    assert L["direction_set_procrustes_r2_ring"] > 0.9
    vp = L["speed_set"]
    mid180 = [c for c in L["chord_speed_readout"] if c["d_theta"] == 180.0][0]
    assert set(L["chord_predictions"]) >= {"mlp_speed_ratio", "nearest_real_same_minus_reduced"}
    assert L["chord_verdict"]["summary"]
    if plane:
        assert vp["better"] == "velocity_plane" and vp["radius_ratio_top_bottom"] > 2.5
        assert mid180["eq9_speed_mean_ratio"] < 0.6                    # the chord drags the speed read down
        assert L["chord_verdict"]["calls"]["eq9_speed_mean_ratio"] == "velocity_plane"
        assert L["chord_verdict"]["calls"]["nearest_real_same_minus_reduced"] == "velocity_plane"
        assert "ring" not in L["chord_verdict"]["calls"].values()
    else:
        assert vp["better"] == "ring" and 0.7 < vp["radius_ratio_top_bottom"] < 1.4
        assert 0.8 < mid180["eq9_speed_mean_ratio"] < 1.2              # a ring leaves the speed read alone
        assert L["chord_verdict"]["calls"]["eq9_speed_mean_ratio"] == "ring"
        assert "velocity_plane" not in L["chord_verdict"]["calls"].values()


def test_geometry_options(tmp_path):
    """--nuisance-regress, --plane chart, and the speed set read as a direction dataset; provenance + layer role."""
    tmp = fake_dataset(tmp_path, "direction")
    run_script("run_geometry_checks.py", tmp, "direction", "--nuisance-regress", "--plane", "chart")
    geo = json.loads((tmp / "results" / "p2_geometry_direction_L1_chart.json").read_text())
    assert geo["subspace"] == "chart" and geo["k"] == 2
    assert geo["angle"]["plane_used"] in ("activation", "centroid") and geo["angle"]["orientation"] in (1, -1)
    nr = geo["nuisance_regressed"]
    assert "speed_mps" not in nr["covariates"] and "motion=velocity" in nr["covariates"]   # speed constant here
    assert nr["angle"]["plane_used"] is not None
    assert geo["layer_role"] == "exploratory" and len(geo["provenance"]["split_sha256"]) == 64
    assert geo["provenance"]["commit"] is not None
    assert "expected_sagitta_by_stride" in geo and geo["summary"]
    sp = tmp_path / "speedset"
    sp.mkdir()
    velocity_dataset(sp, plane=False)
    spec = importlib.util.spec_from_file_location("g", ROOT / "scripts" / "run_geometry_checks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = module.run(module.parse(["--dataset", "speed", "--variable", "direction", "--layer", "1", "--k", "16",
                                   "--act-dir", str(sp / "act"), "--table", str(sp / "table.csv"),
                                   "--split", str(sp / "split.json"), "--results-dir", str(sp / "results"),
                                   "--figures-dir", str(sp / "figures"), "--nuisance-regress"]))
    assert (sp / "results" / "p2_geometry_speed_direction_L1.json").exists()
    assert out["angle"]["plane_used"] is not None
    assert "speed_mps" in out["nuisance_regressed"]["covariates"]


def test_geometry_speed_negative_summary(tmp_path):
    tmp = fake_dataset(tmp_path, "speed")
    run_script("run_geometry_checks.py", tmp, "speed")
    geo = json.loads((tmp / "results" / "p2_geometry_speed_L1.json").read_text())
    assert any(line.startswith("pre-registered negative") for line in geo["summary"])


def test_part2_chart_plane_nuisance(tmp_path):
    tmp = fake_dataset(tmp_path, "direction")
    run_script("run_part2.py", tmp, "direction", "--variable", "direction", "--n-clips", "6", "--K", "5",
               "--n-controls", "2", "--no-bf16", "--plane", "chart", "--nuisance-regress")
    res = json.loads((tmp / "results" / "p2_steer_direction_direction_L1_scattered_chart_nuis.json").read_text())
    assert res["subspace"] == "chart" and res["k"] == 2 and res["nuisance_regressed"]["covariates"]
    assert res["provenance"]["holdout"] == "scattered" and res["layer_role"] == "exploratory"
    man = res["summary"]["manifold"]["overall"]
    assert man["probe_err_to_target"] < 0.3 * man["probe_err_to_true"]      # the edit moves the probe to the target


def test_part2_inlp_plane_nuisance_uses_standardised_basis_as_is(tmp_path, monkeypatch):
    """With --nuisance-regress the activations are already train-standardised, so the INLP Q must span the steering
    plane unchanged (not Q divided by the residuals' SD)."""
    from wm import geometry_checks as gc
    tmp = fake_dataset(tmp_path, "direction")
    X = np.load(tmp / "act" / "meanpool.npy")
    X[1::2, 1, 2:6] += 3.0                            # features the motion covariate explains (residual SD << 1)
    np.save(tmp / "act" / "meanpool.npy", X)
    Q = np.linalg.qr(np.random.default_rng(1).standard_normal((D, 4)))[0]
    np.savez(tmp / "basis.npz", Q=Q)
    seen, real = {}, gc.fit_subspace

    def spy(X, labels, kind="pca", k=64, basis=None):
        seen["B"] = basis
        return real(X, labels, kind, k, basis)

    monkeypatch.setattr(gc, "fit_subspace", spy)
    run_script("run_part2.py", tmp, "direction", "--variable", "direction", "--n-clips", "6", "--K", "5",
               "--n-controls", "2", "--no-bf16", "--plane", "inlp", "--basis", str(tmp / "basis.npz"),
               "--nuisance-regress")
    cos = np.linalg.svd(Q.T @ np.linalg.qr(seen["B"])[0], compute_uv=False)
    assert np.allclose(cos, 1.0, rtol=0, atol=1e-10)


def test_geometry_inlp_plane_nuisance_uses_standardised_basis_as_is(tmp_path, monkeypatch):
    """run_geometry_checks --plane inlp --nuisance-regress: the residual pass gets the stored Q unchanged (it used to
    get basis=None and crash in fit_subspace)."""
    from wm import geometry_checks as gc
    tmp = fake_dataset(tmp_path, "direction")
    Q = np.linalg.qr(np.random.default_rng(1).standard_normal((D, 4)))[0]
    np.savez(tmp / "basis.npz", Q=Q)
    seen, real = [], gc.fit_subspace

    def spy(X, labels, kind="pca", k=64, basis=None):
        seen.append(basis)
        return real(X, labels, kind, k, basis)

    monkeypatch.setattr(gc, "fit_subspace", spy)
    run_script("run_geometry_checks.py", tmp, "direction", "--plane", "inlp", "--basis", str(tmp / "basis.npz"),
               "--nuisance-regress")
    geo = json.loads((tmp / "results" / "p2_geometry_direction_L1_inlp.json").read_text())
    assert "nuisance_regressed" in geo
    cos = np.linalg.svd(Q.T @ np.linalg.qr(seen[-1])[0], compute_uv=False)
    assert np.allclose(cos, 1.0, rtol=0, atol=1e-10)


def test_velocity_chord_uses_knot_folds_only(tmp_path, monkeypatch):
    sp = tmp_path / "speed"
    sp.mkdir()
    velocity_dataset(sp, plane=True)
    spec = importlib.util.spec_from_file_location("vp", ROOT / "scripts" / "run_velocity_plane.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from wm.p2_data import load_inputs
    s = load_inputs("speed", 1, "direction", sp / "act", sp / "table.csv", sp / "split.json")
    seen = []
    real_fit_pca = module.mf.fit_pca
    monkeypatch.setattr(module.mf, "fit_pca", lambda X, k: seen.append(len(X)) or real_fit_pca(X, k))
    module.chord_speed_readout(s, 8, 20, 0)
    assert seen == [int((s["role"] == "knot").sum())]


def test_part2_inlp_plane_refit(tmp_path):
    tmp = fake_dataset(tmp_path, "direction")
    run_script("run_part2.py", tmp, "direction", "--variable", "direction", "--n-clips", "6", "--K", "5",
               "--n-controls", "2", "--no-bf16", "--plane", "inlp", "--max-probe-rounds", "2", "--holdout", "contiguous")
    res = json.loads((tmp / "results" / "p2_steer_direction_direction_L1_contiguous_inlp.json").read_text())
    b = res["subspace_basis"]
    assert b["rows"] == "knot folds at kept values only" and b["folds"] == [0, 1, 2]
    fold = json.loads((tmp / "split.json").read_text())["fold"]
    y = pd.read_csv(tmp / "table.csv")["theta_degrees"].to_numpy()
    held = set(res["held_out_values"])
    expect = sum(1 for i, v in enumerate(y) if 0 <= fold[str(i)] <= 2 and v not in held)
    assert b["n_rows"] == expect                                      # knot rows at the 56 kept values only
    assert res["k"] == 2 * b["n_probes"]


def test_speed_cells_edges_from_fit_rows_only():
    spec = importlib.util.spec_from_file_location("vp", ROOT / "scripts" / "run_velocity_plane.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    speed = np.r_[np.linspace(1, 2, 80), np.full(20, 100.0)]          # extreme speeds only in the non-fit rows
    fit = np.r_[np.ones(80, bool), np.zeros(20, bool)]
    cell, cs = module.speed_direction_cells(speed, np.zeros(100), fit)
    sb = cell.astype(int) // 16
    assert np.bincount(sb[fit], minlength=8).tolist() == [10] * 8       # octiles of the fit rows
    assert max(cs.values()) < 3                                         # bin speeds from fit rows only


def test_position_sheet(tmp_path):
    """A curved sheet (x, y, x^2 + y^2): TPS recovers it; its held-out endpoint beats the flat chord's."""
    rng = np.random.default_rng(4)
    n = 1800
    xy = rng.uniform(-1.2, 1.2, (n, 2))
    X = rng.standard_normal((n, N_LAYERS, D)).astype(np.float32) * 0.2
    X[:, 1, 0], X[:, 1, 1] = 3 * xy[:, 0], 3 * xy[:, 1]
    X[:, 1, 2] = 3 * (xy ** 2).sum(1)
    act = tmp_path / "act"
    act.mkdir()
    np.save(act / "meanpool.npy", X)
    ids = np.arange(n)
    (act / "ids.json").write_text(json.dumps(ids.tolist()))
    pd.DataFrame({"id": ids, "label": 1.0, "theta_degrees": 0.0, "speed_mps": rng.uniform(0.25, 4, n),
                  "acceleration_mps2": 0.0, "motion": "velocity", "start_x": xy[:, 0],
                  "start_y": xy[:, 1]}).to_csv(tmp_path / "table.csv", index=False)
    (tmp_path / "split.json").write_text(json.dumps({"fold": {str(i): (-1 if i % 5 == 0 else int(i % 5 - 1)) for i in ids}}))
    spec = importlib.util.spec_from_file_location("sheet", ROOT / "scripts" / "run_position_sheet.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = module.run(module.parse(["--layer", "1", "--k", "16", "--K", "11", "--n-clips", "20", "--n-controls", "5",
                                   "--act-dir", str(act), "--table", str(tmp_path / "table.csv"),
                                   "--split", str(tmp_path / "split.json"),
                                   "--results-dir", str(tmp_path / "results"), "--figures-dir", str(tmp_path / "figures")]))
    geo = out["geometry_train"]
    assert geo["n_cells"] == 36 and geo["procrustes_r2_centroids_to_xy"] > 0.6
    assert geo["unsupervised_top2_vs_xy_procrustes_r2"] > 0.9
    assert len(out["grid"]["held_out_cells"]) == 4
    s = out["summary"]
    assert out["tps_smoothing_choice"]["best_tps_smoothing"] == out["tps_smoothing"]
    assert out["verdict"]["text"] and "tps_better_on" in out["verdict"]
    assert out["tps_smoothing_choice"]["tps_beats_linear_interp"]                  # the sheet is curved
    assert s["tps"]["excess_to_ref"] < s["chord_linear_interp"]["excess_to_ref"]
    np.testing.assert_allclose(s["tps"]["err_end"], s["chord"]["err_end"], rtol=1e-9)   # endpoint-matched chord
    assert s["tps"]["excess_to_ref"] < s["chord"]["excess_to_ref"]
    assert out["controls"]["random_endpoint_matched"]["excess_to_ref"]["spline_rank"] == 1
    assert (tmp_path / "results" / "p2_sheet_speed_L1.json").exists()
    assert (tmp_path / "figures" / "fig4_sheet_speed_L1.png").exists()


def test_sheet_linear_interp_nan_fallback():
    spec = importlib.util.spec_from_file_location("sheet", ROOT / "scripts" / "run_position_sheet.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    f, count = module.linear_interp(pts, np.arange(4.0)[:, None])
    out = f(np.array([[0.5, 0.5], [3.0, 3.0]]))
    assert np.isfinite(out).all() and out[1, 0] == 3.0 and count["nan_fallbacks"] == 1


def test_sheet_smoothing_never_sees_the_evaluation_block():
    spec = importlib.util.spec_from_file_location("sheet", ROOT / "scripts" / "run_position_sheet.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for G, ev in ((6, 0), (6, 3), (8, 1)):
        seeds, blocks = module.selection_blocks(G, ev)
        evb = set(module.held_block(G, ev))
        assert ev not in seeds and len(blocks) >= 2
        assert all(not (set(b) & evb) for b in blocks) and len({tuple(b) for b in blocks}) == len(blocks)


def test_sheet_smoothing_choice_drops_evaluation_block_rows():
    """The evaluation block's rows must not enter the smoothing choice even as TPS anchors: NaN activations there
    would poison every reconstruction error if they did."""
    spec = importlib.util.spec_from_file_location("sheet", ROOT / "scripts" / "run_position_sheet.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    G, rng = 6, np.random.default_rng(0)
    centres = np.array([[i, j] for j in range(G) for i in range(G)], float)
    cell = np.repeat(np.arange(G * G), 5)
    X = centres[cell] @ rng.standard_normal((2, 8)) + 0.01 * rng.standard_normal((len(cell), 8))
    ev = module.held_block(G, 0)
    X[np.isin(cell, ev)] = np.nan
    seeds, _ = module.selection_blocks(G, 0)
    _, table = module.choose_smoothing(X, cell, centres, G, 3, seeds, exclude=ev)
    assert all(np.isfinite(v) for v in table["mean_error"].values())


def test_summarize_p2_audit_row_and_aggregate():
    spec = importlib.util.spec_from_file_location("sa", ROOT / "scripts" / "summarize_p2_audit.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    ov = lambda e, dn: {"overall": {"probe_err_to_target": e, "delta_norm": dn}}
    r = {"angle_source": "labels_angle", "spline": "smoothing", "holdout": {"block_first_value": 10.0,
         "block_last_value": 50.0}, "held_out_values": [10.0, 50.0],
         "gaps": {"manifold_minus_linear": {"probe_err_to_target": {"mean": 2.0, "ci95": [1, 3]},
                                            "probe_radius_min": {"mean": 0.3, "ci95": [0.2, 0.4]}}},
         "summary": {"manifold": ov(9.0, 4.0), "linear": ov(7.0, 3.0)},
         "verdict": {"call": "negative_endpoint", "sagitta_over_noise_median": 0.4}}
    row = m.run_row(r)
    assert row["curve_extension"] == "cubic" and row["held_out"] == [10.0, 50.0]
    assert (row["endpoint_gap"], row["min_radius_gap"], row["delta_norm_chord"]) == (2.0, 0.3, 3.0)
    a = m.agg([row, {**row, "endpoint_gap": 4.0, "verdict": "positive"}])
    assert a["endpoint_gap_mean"] == 3.0 and a["verdict_counts"] == {"negative_endpoint": 1, "positive": 1}


def test_angle_goodfire_row(tmp_path):
    from wm.p2_data import load_inputs
    fake_dataset(tmp_path, "direction")
    spec = importlib.util.spec_from_file_location("ag", ROOT / "scripts" / "run_angle_goodfire.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    d = load_inputs("direction", 1, "direction", tmp_path / "act", tmp_path / "table.csv", tmp_path / "split.json")
    row = m.angle_row(d, seed=0, k=16)
    assert row["goodfire_passes_periodicity_test"]
    assert abs(row["goodfire_angle_vs_labels"]["circular_corr"]) > 0.95
    assert row["pipeline_source"].startswith("unsupervised_angle")


def test_isometry_path_length_matrix_circle():
    from wm import manifold as mf
    spec = importlib.util.spec_from_file_location("iso", ROOT / "scripts" / "run_isometry_linear.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    v = np.arange(16) * 22.5
    th = np.radians(v)
    cent = {"C": np.stack([2 * np.cos(th), 2 * np.sin(th)], 1), "values": v, "count": np.ones(16),
            "sd_coord": np.ones(2)}
    G = m.path_length_matrix(mf.fit_curve(cent, True, angle="labels"), v, n_steps=150)
    dth = np.radians(np.abs((v[:, None] - v[None] + 180) % 360 - 180))
    np.testing.assert_allclose(G, 2 * dth, rtol=2e-3, atol=1e-9)

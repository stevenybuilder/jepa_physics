"""Disk-token direction edits without the twin's disk location (QA #338), + dose sweep, delta shape, knot order, block.

session2_direction_natural_norm.py disk_forward put the point-12 chord direction on obj = dilate(disk(source) |
disk(twin)) at the twin's per-token change and got 40.8 deg to target (pooled natural-norm chord 79.7, unedited 92.1).
That set contains WHERE THE DISK WILL BE under the target motion (the twin's disk), which no curve arm is given. This
reruns the same forward / predictor / reader on token sets that do not use the twin's disk location:

  token sets (context tubelets 0-3, run_token_patching.token_sets(mask_src, mask_twin))
     union     obj  = dilate(disk(src) | disk(twin))        the stored set (reference; should reproduce 40.8)
     src       objA = dilate(disk(src))                      source disk + 1-patch ring only (the #338 fix)
     twinonly  obj & ~objA                                   the tokens the twin's location ADDS (decomposition)
  dose (per token i, at the edit point L, context-only encoding)
     pertoken  unit(delta) * m * n_i, n_i = ||h_twin[i] - h_src[i]||; n_i is defined on every token, so on the src set
               every token gets its own twin change (no fallback is ever needed); m = 0.5 / 1 / 2
     uniform   unit(delta) * mean_{i in set} n_i on every token of the set: no per-token twin pattern, only one twin-
               derived scalar per (carrier, target)
  directions (unit vectors; per (carrier, target))
     chord12      session-2 chord endpoint at point 12 (natural_norm deltas.npz)
     spline12     the paper's interpolating periodic spline, knots in their label-free (unsupervised-angle, centroid-
                  plane) order: the stored session-2 delta (gave 100.2 deg on the union set)
     splinelab12  the same interpolating periodic spline with LABEL-ORDERED knots (mf.fit_curve angle="labels")
     fourier12    Z ~ a + B [cos t, sin t, cos 2t, sin 2t], OLS on knot rows at kept values in the session-2 PCA-64;
                  straight edit (F(t*) - F(src)) B lifted to 1024-D (run_bakeoff_unified_16arc 'fourier2_4d')
     chord8       session-2 chord at point 8 (plan.npz deltas_L8, 'chord'; that layer's label angle choice)
     chord16/19   chord planned at points 16 / 19 (natural_norm bg_deltas.npz, as the stored disk16/19 arms)
  readers  predictor-native direction probe on the step-mean pooled forecast; err to target (deg, mean over 4 targets
           per carrier) and R; 95% CI = bootstrap over carriers (run_session2.boot_ci, seed 0). Identical to
           session2_direction_natural_norm disk_score / bg_score.

  plan     Mac CPU  -> artifacts/session2/disk_sweep/sweep_deltas.npz + plan_meta.json
  forward  box GPU  -> <root>/sweep_pred.npz + sweep_forward_info.json   (takes /tmp/wm_gpu.lock itself)
  score    Mac CPU  -> results/session2_disk_token_sweep.json + figures/fig_disk_token_sweep.png
"""
import fcntl
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm.data import PROJECT_ROOT, decode, load_table  # noqa: E402

OUT = PROJECT_ROOT / "artifacts" / "session2"
NN = OUT / "natural_norm"
DS = OUT / "disk_sweep"
RES = PROJECT_ROOT / "results"
DIRS = ("chord12", "spline12", "splinelab12", "fourier12", "chord8", "chord16", "chord19")
DIR_POINT = {"chord12": 12, "spline12": 12, "splinelab12": 12, "fourier12": 12, "chord8": 8, "chord16": 16,
             "chord19": 19}
# (name, direction, token set, dose kind, multiplier)
ARMS = (
    ("union_chord_1x", "chord12", "union", "pertoken", 1.0),
    ("src_chord_0.5x", "chord12", "src", "pertoken", 0.5),
    ("src_chord_1x", "chord12", "src", "pertoken", 1.0),
    ("src_chord_2x", "chord12", "src", "pertoken", 2.0),
    ("src_chord_uniform", "chord12", "src", "uniform", 1.0),
    ("twinonly_chord_1x", "chord12", "twinonly", "pertoken", 1.0),
    ("src_fourier4_1x", "fourier12", "src", "pertoken", 1.0),
    ("src_spline_labelfree_1x", "spline12", "src", "pertoken", 1.0),
    ("src_spline_labels_1x", "splinelab12", "src", "pertoken", 1.0),
    ("union_spline_labels_1x", "splinelab12", "union", "pertoken", 1.0),
    ("src_chord_L8_1x", "chord8", "src", "pertoken", 1.0),
    ("src_chord_L16_1x", "chord16", "src", "pertoken", 1.0),
    ("src_chord_L19_1x", "chord19", "src", "pertoken", 1.0),
)
POINTS = (8, 12, 16, 19)
LOCK = "/tmp/wm_gpu.lock"


def arm_point(arm):
    return DIR_POINT[arm[1]]


def token_masks(mask_src, mask_twin):
    """bool [1024] sets for one (carrier, target): union / src / twinonly (see module doc)."""
    from run_token_patching import token_sets
    ts = token_sets(mask_src, mask_twin)
    return {"union": ts["obj"], "src": ts["objA"], "twinonly": ts["obj"] & ~ts["objA"]}


def dose(n_all, mask, kind, mult):
    """Per-token magnitudes [1024] (numpy or torch): n_all = ||h_twin - h_src|| per token."""
    m = n_all * 0
    if mask.sum() == 0:
        return m
    if kind == "pertoken":
        m[mask] = mult * n_all[mask]
    elif kind == "uniform":
        m[mask] = mult * n_all[mask].mean()
    else:
        raise ValueError(kind)
    return m


def plan():
    from wm import manifold as mf
    from wm import motiongeom as mg
    from wm.p2_data import load_inputs
    from run_session2 import load_plan
    arrays, info = load_plan(OUT)
    dz = np.load(NN / "deltas.npz")
    zb = np.load(NN / "bg_deltas.npz")
    idx, carriers, tg = dz["idx"], dz["carrier_rows"], dz["targets"]
    C, T = tg.shape
    held = np.array(info["held_values"])
    out = {"idx": idx, "carrier_rows": carriers, "targets": tg}
    ai = {a: i for i, a in enumerate(info["arms"])}
    out["chord12"] = dz["deltas_L12"][:, 0, 1]                                    # natural_norm ARMS order
    out["spline12"] = dz["deltas_L12"][:, 0, 0]
    out["chord8"] = arrays["deltas_L8"][idx][:, ai["chord"]]
    out["chord16"], out["chord19"] = zb["chord_L16"], zb["chord_L19"]
    d = load_inputs("direction", 12)
    y, X = d["y"], d["X"].astype(np.float64)
    knot = (d["role"] == "knot") & ~np.isin(y, held)
    pca = mf.fit_pca(X[knot], info["k"])
    cent = mf.centroids(pca.project(X[knot]), y[knot])
    choice = info["per_layer"]["12"]["angle_choice"]
    Z, src = pca.project(X[carriers]), y[carriers]
    curves = {"free": mf.fit_curve(cent, True, angle=choice["angle"], plane=choice["plane"], spline=info["spline"]),
              "labels": mf.fit_curve(cent, True, angle="labels", spline=info["spline"])}
    sp = {}
    for nm, cv in curves.items():
        ta = cv.coord_of_value(src)
        sp[nm] = np.stack([pca.lift_delta(mf.manifold_coords(Z, cv, ta, cv.coord_of_value(tg[:, j]), 2)[:, -1] - Z)
                           for j in range(T)], 1)
    out["splinelab12"] = sp["labels"]
    Zk = pca.project(X[knot])
    Fk = mg.fourier_features(y[knot], 2)
    a0, B = mg.fit_linear_encoding(Zk, Fk)
    dF = (mg.fourier_features(tg.reshape(-1), 2) - np.repeat(mg.fourier_features(src, 2), T, 0)).reshape(C, T, 4) @ B
    out["fourier12"] = pca.lift_delta(dF)
    rel = lambda a, b: np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-12)  # noqa
    cos = lambda a, b: float(np.median((a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)))  # noqa
    meta = {"parity_labelfree_spline_rebuild_vs_stored_rel_max": float(rel(sp["free"], out["spline12"]).max()),
            "angle_choice_L12_labelfree": choice,
            "fourier_fit_r2_pca64_knot_rows": float(1 - ((Zk - a0 - Fk @ B) ** 2).sum() / ((Zk - Zk.mean(0)) ** 2).sum()),
            "n_knot_rows": int(knot.sum()), "k": info["k"], "held_values": held.tolist(),
            "own_norm_median": {k: float(np.median(np.linalg.norm(out[k], axis=-1))) for k in DIRS},
            "cos_median": {"splinelab_vs_splinefree": cos(sp["labels"], out["spline12"]),
                           "splinelab_vs_chord": cos(sp["labels"], out["chord12"]),
                           "splinefree_vs_chord": cos(out["spline12"], out["chord12"]),
                           "fourier_vs_chord": cos(out["fourier12"], out["chord12"])}}
    f4 = OUT / "fourier4" / "plan.npz"
    if f4.exists():
        zf = np.load(f4)
        if "t1_L12" in zf.files and (zf["idx"] == idx).all():
            meta["parity_fourier_vs_session2_fourier4_plan_rel_max"] = float(rel(out["fourier12"], zf["t1_L12"][:, 0]).max())
    for k in DIRS:
        out[k] = np.asarray(out[k], np.float32)
        assert out[k].shape == (C, T, 1024), (k, out[k].shape)
    DS.mkdir(parents=True, exist_ok=True)
    np.savez(DS / "sweep_deltas.npz", **out)
    (DS / "plan_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def forward(root, n_carriers=200, arms=None, batch_size=12):
    """arms: comma-separated arm names (a shard); default all. Output <root>/sweep_pred.npz holds only those arms."""
    import torch
    torch.set_num_threads(4)
    from wm.data import disk_mask
    from wm.extract import load_model, pick_device, preprocess, set_precision
    from wm.predictor_readout import pool_steps, predict_future
    from wm.propagate import prefix
    root = Path(root)
    lockf = open(LOCK, "a")
    tw0 = time.time()
    fcntl.flock(lockf, fcntl.LOCK_EX)
    lock_wait = time.time() - tw0
    print(f"lock acquired after {lock_wait:.0f}s", flush=True)
    set_precision()
    device = pick_device()
    model = load_model("vjepa2", device)
    enc = model.encoder
    z = np.load(root / "sweep_deltas.npz")
    idx, rows = z["idx"][:n_carriers], z["carrier_rows"][:n_carriers]
    C, T = len(rows), z["targets"].shape[1]
    unit = {}
    for k in DIRS:
        v = z[k][:C].astype(np.float64)
        unit[k] = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    sel = [a for a in ARMS if arms is None or a[0] in arms.split(",")]
    assert sel and (arms is None or len(sel) == len(arms.split(","))), arms
    pts = sorted({DIR_POINT[a[1]] for a in sel})
    A = len(sel)
    df = load_table("direction")
    pred = np.zeros((C, A, T, 4, 1024), np.float32)
    ntok = np.zeros((C, A, T), np.int32)
    energy = np.zeros((C, A, T), np.float32)
    dose_mean = np.zeros((C, A, T), np.float32)          # mean magnitude over edited tokens
    dose_median = np.zeros((C, A, T), np.float32)
    mp_shift = np.zeros((C, A, T), np.float32)           # ||mean over all 1024 tokens of the edit||
    src_pp = np.zeros((C, 4, 1024), np.float32)
    t0 = time.time()

    @torch.no_grad()
    def run(h, L):
        for layer in enc.layer[L:]:
            h = layer(h, None, None, False)[0]
        return pool_steps(predict_future(model, enc.layernorm(h), 0))

    for c in range(C):
        fs = decode(df["video"].iloc[rows[c]])
        ft = [decode(root / "twins" / f"twin_{int(idx[c]):04d}_{j}.mp4") for j in range(T)]
        ms = disk_mask(fs)[0]
        pv = preprocess([fs] + ft).to(device)
        saved, fin0, _ = prefix(model, pv[:, :8], pts)
        src_pp[c] = pool_steps(predict_future(model, fin0[:1], 0)).cpu().numpy()[0]
        jobs = []
        for j in range(T):
            masks = {k: torch.as_tensor(v, device=device) for k, v in token_masks(ms, disk_mask(ft[j])[0]).items()}
            for ai, (name, dk, sk, kind, mult) in enumerate(sel):
                L = DIR_POINT[dk]
                h = saved[L]
                n_all = (h[1 + j] - h[0]).float().norm(dim=-1)
                mag = dose(n_all, masks[sk], kind, mult)
                nz = mag > 0
                ntok[c, ai, j] = int(nz.sum())
                energy[c, ai, j] = float((mag ** 2).sum())
                dose_mean[c, ai, j] = float(mag[nz].mean()) if nz.any() else 0.0
                dose_median[c, ai, j] = float(mag[nz].median()) if nz.any() else 0.0
                mp_shift[c, ai, j] = float(mag.mean())            # |mean_tok(mag_i u)| = mean(mag) since u is shared
                u = torch.as_tensor(unit[dk][c, j], dtype=h.dtype, device=device)
                jobs.append((ai, j, L, mag.to(h.dtype)[:, None] * u[None, :]))
        for L in pts:
            jl = [q for q in jobs if q[2] == L]
            for b in range(0, len(jl), batch_size):
                chk = jl[b:b + batch_size]
                p = run(saved[L][:1] + torch.stack([q[3] for q in chk]), L).cpu().numpy()
                for q, (ai, j, _, _) in enumerate(chk):
                    pred[c, ai, j] = p[q]
        if c % 10 == 0 or c == C - 1:
            print(f"{c + 1}/{C} carriers {time.time() - t0:.0f}s", flush=True)
    secs = time.time() - t0
    np.savez(root / "sweep_pred.npz", idx=idx, arms=np.array([a[0] for a in sel]), pred=pred, ntok=ntok, energy=energy, dose_mean=dose_mean,
             dose_median=dose_median, mp_shift=mp_shift, src_pred_pooled=src_pp)
    info = {"seconds": secs, "lock_wait_seconds": lock_wait, "n_carriers": C, "arms": [a[0] for a in sel],
            "arm_spec": [list(a) for a in sel], "points": pts,
            "deltas_sha256": hashlib.sha256((root / "sweep_deltas.npz").read_bytes()).hexdigest(), "batch_size": batch_size, "torch_threads": 4,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "torch": torch.__version__}
    (root / "sweep_forward_info.json").write_text(json.dumps(info, indent=1))
    fcntl.flock(lockf, fcntl.LOCK_UN)
    print(json.dumps(info))


def merge_shards(sdir=None):
    """Assemble artifacts/session2/disk_sweep/shards/<shard>/sweep_pred.npz (+ sweep_forward_info.json, box.json) into
    full [C, len(ARMS), ...] arrays in ARMS order. Every arm must come from exactly one shard; idx must agree."""
    sdir = Path(sdir) if sdir else DS / "shards"
    parts, infos = {}, {}
    for d in sorted(p for p in sdir.iterdir() if (p / "sweep_pred.npz").exists()):
        zz = np.load(d / "sweep_pred.npz")
        inf = json.loads((d / "sweep_forward_info.json").read_text())
        bx = d / "box.json"
        inf["box"] = json.loads(bx.read_text()) if bx.exists() else None
        infos[d.name] = inf
        for ai, a in enumerate(zz["arms"].tolist()):
            assert a not in parts, f"arm {a} in two shards"
            parts[a] = (d.name, ai, zz)
    missing = [a[0] for a in ARMS if a[0] not in parts]
    assert not missing, f"missing arms {missing}"
    idx0 = next(iter(parts.values()))[2]["idx"]
    out = {"idx": idx0}
    for k in ("pred", "ntok", "energy", "dose_mean", "dose_median", "mp_shift"):
        out[k] = np.stack([parts[a[0]][2][k][:, parts[a[0]][1]] for a in ARMS], 1)
    for a in ARMS:
        assert (parts[a[0]][2]["idx"] == idx0).all()
    src = [p[2]["src_pred_pooled"] for p in parts.values()]
    out["src_pred_pooled"] = src[0]
    out["src_pred_pooled_cross_shard_rel_maxabs"] = float(max(np.abs(s - src[0]).max() for s in src) / np.abs(src[0]).max())
    finfo = {"n_carriers": int(len(idx0)), "arms": [a[0] for a in ARMS], "shards": infos,
             "arm_to_shard": {a[0]: parts[a[0]][0] for a in ARMS},
             "gpu_seconds_total": float(sum(i["seconds"] for i in infos.values())),
             "src_pred_pooled_cross_shard_rel_maxabs": out["src_pred_pooled_cross_shard_rel_maxabs"]}
    return out, finfo


def score(box=None):
    from run_session2 import angle_of, boot_ci, load_groups, load_plan, sha256_file, wrap, write
    from wm.p2_data import load_inputs
    from wm.probes import Standardizer, fit_ridge, predict, targets
    nat_json = json.loads((RES / "session2_predictor_native_readout.json").read_text())
    df = load_table("direction")
    y = df["theta_degrees"].to_numpy(float)
    Y = targets(df, "direction")[0]
    probe = load_inputs("direction", 0)["role"] == "probe"
    Zall = np.load(OUT / "native" / "pred_pooled_all.npy").astype(np.float64)
    st = Standardizer().fit(Zall.mean(1)[probe])
    W, b = fit_ridge(st.transform(Zall.mean(1)[probe]), Y[probe],
                     nat_json["direction"]["predictor_output_probe_stepmean"]["alpha"])
    readP = lambda X: predict(st.transform(X.reshape(-1, X.shape[-1])), W, b).reshape(X.shape[:-1] + (-1,))  # noqa
    arrays, info = load_plan(OUT)
    F = load_groups(OUT / "forward", keys={"idx", "src_pred_pooled"})
    z, finfo = merge_shards()
    meta = json.loads((DS / "plan_meta.json").read_text())
    C = finfo["n_carriers"]
    assert (z["idx"] == F["idx"][:C]).all()
    rows, tg = arrays["carrier_rows"][F["idx"][:C]], arrays["targets"][F["idx"][:C]]
    u = lambda th: np.stack([np.sin(np.radians(th)), np.cos(np.radians(th))], -1)  # noqa
    d_true = u(tg) - u(y[rows])[:, None]
    proj = lambda a, bb: (a * bb).sum(-1) / np.maximum((bb * bb).sum(-1), 1e-9)  # noqa
    Ps = readP(F["src_pred_pooled"][:C].astype(np.float64).mean(1))
    pooled = json.loads((RES / "session2_direction_natural_norm.json").read_text())
    dk = json.loads((RES / "session2_direction_natural_norm_disktokens.json").read_text())
    bgc = json.loads((RES / "session2_direction_natural_norm_bgcontrol.json").read_text())
    nat12 = np.load(NN / "deltas.npz")["natural_norm_L12"][:C].astype(np.float64)       # [C, T] pooled natural norm
    src_par = float(np.abs(z["src_pred_pooled"] - F["src_pred_pooled"][:C]).max() / np.abs(F["src_pred_pooled"]).max())
    errs, table = {}, {}
    for ai, (name, dkey, sk, kind, mult) in enumerate(ARMS):
        Pe = readP(z["pred"][:, ai].astype(np.float64).mean(2))
        err, R = wrap(angle_of(Pe) - tg).mean(1), proj(Pe - Ps[:, None], d_true).mean(1)
        errs[name] = err
        nt = z["ntok"][:, ai]
        table[name] = {"direction": dkey, "point": DIR_POINT[dkey], "token_set": sk, "dose": kind, "multiplier": mult,
                       "err_to_target": boot_ci(err), "R": boot_ci(R),
                       "edited_tokens": {"median": float(np.median(nt)), "mean": float(nt.mean()),
                                         "min": int(nt.min()), "max": int(nt.max()),
                                         "frac_pairs_empty": float((nt == 0).mean())},
                       "per_token_dose_mean_median": float(np.median(z["dose_mean"][:, ai][nt > 0])) if (nt > 0).any() else 0.0,
                       "per_token_dose_median_median": float(np.median(z["dose_median"][:, ai][nt > 0])) if (nt > 0).any() else 0.0,
                       "edit_energy_median": float(np.median(z["energy"][:, ai])),
                       "meanpool_shift_over_pooled_natural_median": float(np.median(z["mp_shift"][:, ai] / nat12))
                       if DIR_POINT[dkey] == 12 else None}
    paired = {"src_chord_1x_minus_union_chord_1x": boot_ci(errs["src_chord_1x"] - errs["union_chord_1x"]),
              "src_chord_2x_minus_src_chord_1x": boot_ci(errs["src_chord_2x"] - errs["src_chord_1x"]),
              "src_chord_1x_minus_src_chord_0.5x": boot_ci(errs["src_chord_1x"] - errs["src_chord_0.5x"]),
              "src_chord_uniform_minus_src_chord_1x": boot_ci(errs["src_chord_uniform"] - errs["src_chord_1x"]),
              "twinonly_minus_src_chord_1x": boot_ci(errs["twinonly_chord_1x"] - errs["src_chord_1x"]),
              "src_fourier4_minus_src_chord_1x": boot_ci(errs["src_fourier4_1x"] - errs["src_chord_1x"]),
              "src_spline_labels_minus_labelfree": boot_ci(errs["src_spline_labels_1x"] - errs["src_spline_labelfree_1x"]),
              "src_spline_labels_minus_src_chord_1x": boot_ci(errs["src_spline_labels_1x"] - errs["src_chord_1x"]),
              "src_chord_L8_minus_L12": boot_ci(errs["src_chord_L8_1x"] - errs["src_chord_1x"]),
              "src_chord_L16_minus_L12": boot_ci(errs["src_chord_L16_1x"] - errs["src_chord_1x"]),
              "src_chord_L19_minus_L12": boot_ci(errs["src_chord_L19_1x"] - errs["src_chord_1x"])}
    stored_disk_chord = dk["table"]["chord"]["err_to_target"]
    dpred = np.load(NN / "disk_pred.npz")["pred"][:C, 1]                              # stored disk chord
    Pd = readP(dpred.astype(np.float64).mean(2))
    Pu = readP(z["pred"][:, 0].astype(np.float64).mean(2))
    res = {"question": "QA #338: is the disk-token 40.8 deg a location oracle (the token set includes the target twin's "
                       "disk)? Source-disk-only token set, dose sweep, delta shape (Fourier-4, label-ordered spline), "
                       "edit block 8/12/16/19; same carriers, targets, forward, predictor and reader as the stored arms",
           "n_carriers": int(C), "n_targets": int(tg.shape[1]), "arms": [a[0] for a in ARMS],
           "definitions": {
               "token_sets": {"union": "dilate(disk(src) | disk(twin)), context tubelets 0-3 (stored disk-token set)",
                              "src": "dilate(disk(src)) only: source disk + 1-patch ring (no twin location)",
                              "twinonly": "union & ~src: tokens the twin's disk location adds"},
               "dose": {"pertoken": "unit(delta) * multiplier * ||h_twin[i] - h_src[i]|| at the edit point, per token "
                                    "(defined on every token, so no fallback dose is needed on the src set)",
                        "uniform": "unit(delta) * mean over the set of ||h_twin[i] - h_src[i]||, same on every token"},
               "directions": {"chord12": "session-2 chord, point 12", "spline12": "paper's interpolating periodic spline, "
                              "label-free knot order (stored; unsupervised angle, centroid plane)",
                              "splinelab12": "same spline, label-ordered knots", "fourier12": "(F(t*)-F(src)) B, OLS "
                              "on knot rows at kept values, PCA-64, lifted", "chord8": "session-2 chord, point 8",
                              "chord16": "chord planned at 16 (bg_deltas.npz)", "chord19": "chord planned at 19"},
               "err_to_target": "as session2_direction_natural_norm.json (predictor-native probe, step-mean forecast, "
                                "mean over 4 targets per carrier); CI = bootstrap over carriers, 1000 draws, seed 0",
               "R": "as session2_direction_natural_norm.json",
               "meanpool_shift_over_pooled_natural": "||mean over 1024 context tokens of the edit|| / ||twin_mp12 - "
                                                     "src_mp12|| (pooled natural norm), median over (carrier, target)"},
           "plan": meta, "forward": finfo, "box": box,
           "parity_src_forecast_vs_session2_cache_rel_maxabs": src_par,
           "union_reproduction": {"stored_err_mean": stored_disk_chord["mean"], "rerun_err_mean": table["union_chord_1x"]["err_to_target"]["mean"],
                                  "maxabs_forecast_angle_diff_deg": float(np.abs(wrap(angle_of(Pu) - angle_of(Pd))).max()),
                                  "median_forecast_angle_diff_deg": float(np.median(wrap(angle_of(Pu) - angle_of(Pd))))},
           "references": {"unedited_err": pooled["unedited"]["err_to_target"], "twin_ceiling": pooled["twin_ceiling"],
                          "pooled_natural_chord_L12": pooled["table"]["12"]["natural"]["chord"],
                          "pooled_natural_spline_L12": pooled["table"]["12"]["natural"]["spline"],
                          "stored_disk_union_chord_L12": dk["table"]["chord"], "stored_disk_union_spline_L12": dk["table"]["spline"],
                          "stored_disk16_union_chord": bgc["table"]["disk16"], "stored_disk19_union_chord": bgc["table"]["disk19"],
                          "stored_bg_count": bgc["table"]["bg_count"], "stored_bg_energy": bgc["table"]["bg_energy"]},
           "table": table, "paired": paired}
    write(RES / "session2_disk_token_sweep.json", res, stage="disk_token_sweep_score", seeds={"bootstrap": 0, "plan": 0},
          caches={k: {"path": str(DS / "shards" / k / "sweep_pred.npz"),
                      "sha256": sha256_file(DS / "shards" / k / "sweep_pred.npz")} for k in finfo["shards"]},
          script_sha256=sha256_file(Path(__file__)))
    for name in res["arms"]:
        t = table[name]
        print(f"{name:26s} err {t['err_to_target']['mean']:6.1f} {np.round(t['err_to_target']['ci95'], 1)}  "
              f"R {t['R']['mean']:.3f} {np.round(t['R']['ci95'], 3)}  ntok {t['edited_tokens']['median']:.0f}  "
              f"dose {t['per_token_dose_mean_median']:.2f}")
    print("union repro", res["union_reproduction"])
    figure(res)


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = res["table"]
    groups = [("Token set (point 12, chord, 1x)", ["union_chord_1x", "src_chord_1x", "twinonly_chord_1x", "src_chord_uniform"],
               ["union\n(src|twin)", "src only", "twin-only\ntokens", "src only\nuniform"]),
              ("Dose (src only, chord, point 12)", ["src_chord_0.5x", "src_chord_1x", "src_chord_2x"], ["0.5x", "1x", "2x"]),
              ("Delta shape (src only, 1x, point 12)", ["src_chord_1x", "src_fourier4_1x", "src_spline_labelfree_1x",
                                                         "src_spline_labels_1x", "union_spline_labels_1x"],
               ["chord", "Fourier-4", "spline\nlabel-free", "spline\nlabel-ord.", "spline lab.\n(union)"]),
              ("Edit block (src only, chord, 1x)", ["src_chord_L8_1x", "src_chord_1x", "src_chord_L16_1x", "src_chord_L19_1x"],
               ["8", "12", "16", "19"])]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.8), gridspec_kw={"width_ratios": [4, 3, 5, 4]})
    une, twin = res["references"]["unedited_err"]["mean"], res["references"]["twin_ceiling"]["err_to_target"]["mean"]
    for a, (title, arms, labels) in zip(ax, groups):
        for x, arm in enumerate(arms):
            r = T[arm]["err_to_target"]
            col = "#d9822b" if T[arm]["token_set"] == "union" else ("#999999" if T[arm]["token_set"] == "twinonly" else "#3b6fb6")
            a.bar(x, r["mean"], color=col, width=0.75)
            a.errorbar(x, r["mean"], yerr=[[r["mean"] - r["ci95"][0]], [r["ci95"][1] - r["mean"]]], color="k", lw=1)
            a.text(x, 4, f"R {T[arm]['R']['mean']:.2f}", ha="center", fontsize=8, color="white", rotation=90)
        a.axhline(une, ls="--", color="k", lw=1, label=f"unedited {une:.1f}")
        a.axhline(twin, ls=":", color="g", lw=1.5, label=f"twin {twin:.1f}")
        a.set_xticks(range(len(arms)))
        a.set_xticklabels(labels, fontsize=8)
        a.set_ylim(0, 115)
        a.set_title(title, fontsize=10)
    ax[0].set_ylabel("forecast direction error to target (deg)")
    ax[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Disk-token direction edits without the twin's disk location (QA #338); {res['n_carriers']} carriers x "
                 f"{res['n_targets']} held-out targets, 95% bootstrap CI over carriers; orange = stored union set", fontsize=10)
    fig.tight_layout()
    out = PROJECT_ROOT / "figures" / "fig_disk_token_sweep.png"
    fig.savefig(out, dpi=140)
    print(out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "plan":
        plan()
    elif cmd == "forward":
        forward(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 200, sys.argv[4] if len(sys.argv) > 4 else None)
    elif cmd == "score":
        score(sys.argv[2] if len(sys.argv) > 2 else None)

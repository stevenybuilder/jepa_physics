"""Twin-free per-token dose for the source-disk chord edit (QA #362).

session2_disk_token_sweep.py (results/session2_disk_token_sweep.json) edits the source-disk token set (dilate(disk(src)),
context tubelets 0-3) at point 12 along unit(chord12) with a per-token dose n_i = ||h_twin[i] - h_src[i]|| (44.7 deg to
target) or a uniform dose mean_{set} n_i (65.8 deg). The per-token profile comes from the carrier's own twin, so the
twin's geometry (where its disk goes) can enter through the dose. This adds twin-free dose profiles:

  p_bin[i] = mean over probe-split clips k with theta_k == target (7-8 clips; carriers are test-split rows, never in the
             probe split) of ||h_k[i] - h_src[i]||  (context-only encoding, point 12; never the carrier's twin)
  p_all[i] = the same mean over all 480 probe-split clips (no target information at all)
Arms (all: direction unit(chord12), token set src, point 12, same forward / predictor / reader as the stored arms)
  src_chord_1x              per-token twin dose (rerun of the stored arm; parity + pairing)
  src_chord_uniform         uniform dose mean_{set} n_i (rerun)
  src_chord_twinfree_bin    p_bin rescaled so its set mean = mean_{set} n_i (the uniform arm's one twin scalar):
                            twin-free SHAPE at the same total dose as the uniform and per-token arms
  src_chord_twinfree_all    p_all rescaled the same way
  src_chord_twinfree_bin_raw  p_bin as is (twin-free shape AND magnitude)
Readers: predictor-native direction probe on the step-mean pooled forecast, err to target (deg, mean over the 4 targets
per carrier), R; CI = bootstrap over carriers (run_session2.boot_ci, 1000, seed 0), paired differences likewise.

  plan     Mac CPU  -> artifacts/session2/disk_sweep_twinfree/twinfree_plan.npz (probe rows, probe angles)
  forward  box GPU  -> <root>/twinfree_pred.npz + twinfree_forward_info.json (takes /tmp/wm_gpu.lock itself)
  score    Mac CPU  -> results/session2_disk_token_sweep_twinfree.json
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
from session2_disk_token_sweep import DS, LOCK, OUT, RES, token_masks  # noqa: E402
from wm.data import PROJECT_ROOT, decode, load_table  # noqa: E402

TF = OUT / "disk_sweep_twinfree"
POINT = 12
# (name, dose kind)
ARMS = (("src_chord_1x", "pertoken"), ("src_chord_uniform", "uniform"), ("src_chord_twinfree_bin", "bin_matched"),
        ("src_chord_twinfree_all", "all_matched"), ("src_chord_twinfree_bin_raw", "bin_raw"))


# ================================================================ pure helpers (tests/test_disk_token_sweep_twinfree.py)

def bin_members(y_probe, target, tol=1e-6):
    """Indices of probe clips whose angle equals the target (circular, within tol degrees)."""
    d = np.abs((np.asarray(y_probe, float) - target + 180.0) % 360.0 - 180.0)
    return np.where(d <= tol)[0]


def twinfree_dose(profile, n_twin, mask, kind):
    """Per-token magnitudes [N] on mask (numpy or torch). kind:
    pertoken  n_twin on mask; uniform  mean_mask(n_twin) on mask;
    matched   profile * mean_mask(n_twin) / mean_mask(profile) (twin-free shape, the uniform arm's scalar);
    raw       profile on mask."""
    m = n_twin * 0
    if mask.sum() == 0:
        return m
    if kind == "pertoken":
        m[mask] = n_twin[mask]
    elif kind == "uniform":
        m[mask] = n_twin[mask].mean()
    elif kind == "matched":
        m[mask] = profile[mask] * (n_twin[mask].mean() / profile[mask].mean())
    elif kind == "raw":
        m[mask] = profile[mask]
    else:
        raise ValueError(kind)
    return m


# ================================================================ plan (Mac)

def plan():
    from wm.p2_data import load_inputs
    df = load_table("direction")
    role = load_inputs("direction", 0)["role"]
    rows = np.where(role == "probe")[0]
    y = df["theta_degrees"].to_numpy(float)
    z = np.load(DS / "sweep_deltas.npz")
    assert not np.isin(z["carrier_rows"], rows).any(), "a carrier is a probe clip"
    counts = {float(t): int(len(bin_members(y[rows], t))) for t in np.unique(z["targets"])}
    TF.mkdir(parents=True, exist_ok=True)
    np.savez(TF / "twinfree_plan.npz", probe_rows=rows, probe_y=y[rows])
    print("probe clips", len(rows), "per target bin", counts)


# ================================================================ forward (box GPU)

def forward(root, n_carriers=200, batch_size=12):
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
    pl = np.load(root / "twinfree_plan.npz")
    idx, rows = z["idx"][:n_carriers], z["carrier_rows"][:n_carriers]
    tg = z["targets"][:n_carriers]
    C, T = len(rows), tg.shape[1]
    v = z["chord12"][:C].astype(np.float64)
    unit = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    df = load_table("direction")
    t0 = time.time()
    # probe-split clips: context-only point-12 tokens, fp16 on the GPU
    prow, py = pl["probe_rows"], pl["probe_y"]
    Hp = torch.empty((len(prow), 1024, 1024), dtype=torch.float16, device=device)
    with torch.no_grad():
        for s in range(0, len(prow), 8):
            pv = preprocess([decode(df["video"].iloc[r]) for r in prow[s:s + 8]]).to(device)
            saved, _, _ = prefix(model, pv[:, :8], [POINT])
            Hp[s:s + len(pv)] = saved[POINT].half()
    t_probe = time.time() - t0
    print(f"probe encodings {len(prow)} in {t_probe:.0f}s", flush=True)
    bins = {float(t): torch.as_tensor(bin_members(py, t), device=device) for t in np.unique(tg)}
    A = len(ARMS)
    pred = np.zeros((C, A, T, 4, 1024), np.float32)
    ntok = np.zeros((C, A, T), np.int32)
    energy = np.zeros((C, A, T), np.float32)
    dose_mean = np.zeros((C, A, T), np.float32)
    mp_shift = np.zeros((C, A, T), np.float32)
    corr_profile_twin = np.full((C, 2, T), np.nan, np.float32)    # Pearson over set tokens: p_bin / p_all vs n_twin
    src_pp = np.zeros((C, 4, 1024), np.float32)

    @torch.no_grad()
    def run(h, L):
        for layer in enc.layer[L:]:
            h = layer(h, None, None, False)[0]
        return pool_steps(predict_future(model, enc.layernorm(h), 0))

    def pearson(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / max(float(a.norm() * b.norm()), 1e-12))

    with torch.no_grad():
        for c in range(C):
            fs = decode(df["video"].iloc[rows[c]])
            ft = [decode(root / "twins" / f"twin_{int(idx[c]):04d}_{j}.mp4") for j in range(T)]
            ms = disk_mask(fs)[0]
            pv = preprocess([fs] + ft).to(device)
            saved, fin0, _ = prefix(model, pv[:, :8], [POINT])
            src_pp[c] = pool_steps(predict_future(model, fin0[:1], 0)).cpu().numpy()[0]
            h = saved[POINT]
            src_set = torch.as_tensor(token_masks(ms, disk_mask(ft[0])[0])["src"], device=device)
            ti = torch.where(src_set)[0]
            dist = (Hp[:, ti].float() - h[0, ti][None]).norm(dim=-1)            # [n_probe, |set|]
            p_all = torch.zeros(1024, device=device)
            p_all[ti] = dist.mean(0)
            jobs = []
            for j in range(T):
                sset = torch.as_tensor(token_masks(ms, disk_mask(ft[j])[0])["src"], device=device)
                assert bool((sset == src_set).all())
                p_bin = torch.zeros(1024, device=device)
                p_bin[ti] = dist[bins[float(tg[c, j])]].mean(0)
                n_all = (h[1 + j] - h[0]).float().norm(dim=-1)
                corr_profile_twin[c, 0, j] = pearson(p_bin[ti], n_all[ti])
                corr_profile_twin[c, 1, j] = pearson(p_all[ti], n_all[ti])
                for ai, (name, kind) in enumerate(ARMS):
                    prof = p_all if kind.startswith("all") else p_bin
                    k2 = kind.split("_")[-1] if "_" in kind else kind
                    mag = twinfree_dose(prof, n_all, src_set, k2)
                    nz = mag > 0
                    ntok[c, ai, j] = int(nz.sum())
                    energy[c, ai, j] = float((mag ** 2).sum())
                    dose_mean[c, ai, j] = float(mag[nz].mean()) if nz.any() else 0.0
                    mp_shift[c, ai, j] = float(mag.mean())
                    u = torch.as_tensor(unit[c, j], dtype=h.dtype, device=device)
                    jobs.append((ai, j, mag.to(h.dtype)[:, None] * u[None, :]))
            for b in range(0, len(jobs), batch_size):
                chk = jobs[b:b + batch_size]
                p = run(h[:1] + torch.stack([q[2] for q in chk]), POINT).cpu().numpy()
                for q, (ai, j, _) in enumerate(chk):
                    pred[c, ai, j] = p[q]
            if c % 10 == 0 or c == C - 1:
                print(f"{c + 1}/{C} carriers {time.time() - t0:.0f}s", flush=True)
    secs = time.time() - t0
    np.savez(root / "twinfree_pred.npz", idx=idx, arms=np.array([a[0] for a in ARMS]), pred=pred, ntok=ntok,
             energy=energy, dose_mean=dose_mean, mp_shift=mp_shift, corr_profile_twin=corr_profile_twin,
             src_pred_pooled=src_pp)
    info = {"seconds": secs, "seconds_probe_encode": t_probe, "lock_wait_seconds": lock_wait, "n_carriers": C,
            "arms": [list(a) for a in ARMS], "point": POINT, "n_probe": int(len(prow)),
            "bin_sizes": {str(k): int(len(v)) for k, v in bins.items()}, "probe_cache_dtype": "float16",
            "deltas_sha256": hashlib.sha256((root / "sweep_deltas.npz").read_bytes()).hexdigest(),
            "plan_sha256": hashlib.sha256((root / "twinfree_plan.npz").read_bytes()).hexdigest(),
            "batch_size": batch_size, "torch_threads": 4,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "torch": torch.__version__}
    (root / "twinfree_forward_info.json").write_text(json.dumps(info, indent=1))
    fcntl.flock(lockf, fcntl.LOCK_UN)
    print(json.dumps(info))


# ================================================================ score (Mac CPU)

def score(root, smoke=False):
    from run_session2 import angle_of, boot_ci, load_groups, load_plan, sha256_file, wrap, write
    from session2_disk_token_sweep import ARMS as S_ARMS, merge_shards
    from wm.p2_data import load_inputs
    from wm.probes import Standardizer, fit_ridge, predict, targets
    root = Path(root)
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
    arrays, _ = load_plan(OUT)
    F = load_groups(OUT / "forward", keys={"idx", "src_pred_pooled"})
    z = np.load(root / "twinfree_pred.npz")
    finfo = json.loads((root / "twinfree_forward_info.json").read_text())
    C = finfo["n_carriers"]
    assert (z["idx"] == F["idx"][:C]).all()
    rows, tg = arrays["carrier_rows"][F["idx"][:C]], arrays["targets"][F["idx"][:C]]
    u = lambda th: np.stack([np.sin(np.radians(th)), np.cos(np.radians(th))], -1)  # noqa
    d_true = u(tg) - u(y[rows])[:, None]
    proj = lambda a, bb: (a * bb).sum(-1) / np.maximum((bb * bb).sum(-1), 1e-9)  # noqa
    Ps = readP(F["src_pred_pooled"][:C].astype(np.float64).mean(1))
    errs, Rs, table = {}, {}, {}
    for ai, (name, kind) in enumerate(ARMS):
        Pe = readP(z["pred"][:, ai].astype(np.float64).mean(2))
        errs[name] = wrap(angle_of(Pe) - tg).mean(1)
        Rs[name] = proj(Pe - Ps[:, None], d_true).mean(1)
        nt = z["ntok"][:, ai]
        table[name] = {"dose": kind, "direction": "chord12", "token_set": "src", "point": POINT,
                       "err_to_target": boot_ci(errs[name]), "R": boot_ci(Rs[name]),
                       "edited_tokens_median": float(np.median(nt)),
                       "per_token_dose_mean_median": float(np.median(z["dose_mean"][:, ai][nt > 0])),
                       "edit_energy_median": float(np.median(z["energy"][:, ai])),
                       "meanpool_shift_median": float(np.median(z["mp_shift"][:, ai]))}
    e = errs
    paired = {"twinfree_bin_minus_pertoken": boot_ci(e["src_chord_twinfree_bin"] - e["src_chord_1x"]),
              "twinfree_bin_minus_uniform": boot_ci(e["src_chord_twinfree_bin"] - e["src_chord_uniform"]),
              "twinfree_all_minus_pertoken": boot_ci(e["src_chord_twinfree_all"] - e["src_chord_1x"]),
              "twinfree_all_minus_uniform": boot_ci(e["src_chord_twinfree_all"] - e["src_chord_uniform"]),
              "twinfree_bin_minus_twinfree_all": boot_ci(e["src_chord_twinfree_bin"] - e["src_chord_twinfree_all"]),
              "twinfree_bin_raw_minus_pertoken": boot_ci(e["src_chord_twinfree_bin_raw"] - e["src_chord_1x"]),
              "twinfree_bin_raw_minus_uniform": boot_ci(e["src_chord_twinfree_bin_raw"] - e["src_chord_uniform"]),
              "uniform_minus_pertoken": boot_ci(e["src_chord_uniform"] - e["src_chord_1x"])}
    frac_gap = {k: None for k in ("bin", "all")}
    for k in frac_gap:   # share of the uniform -> per-token improvement the twin-free shape recovers
        num, den = e["src_chord_uniform"] - e[f"src_chord_twinfree_{k}"], e["src_chord_uniform"] - e["src_chord_1x"]
        ix = np.random.default_rng(0).integers(0, C, (1000, C))
        r = num[ix].mean(1) / den[ix].mean(1)
        frac_gap[k] = {"ratio": float(num.mean() / den.mean()), "ci95": [float(np.percentile(r, 2.5)),
                                                                         float(np.percentile(r, 97.5))]}
    # parity with the stored sweep (same arms, same carriers)
    stored = json.loads((RES / "session2_disk_token_sweep.json").read_text())
    parity = {}
    if not smoke:
        zs, sinfo = merge_shards()
        names = [a[0] for a in S_ARMS]
        for nm in ("src_chord_1x", "src_chord_uniform"):
            a_new, a_old = z["pred"][:, [a[0] for a in ARMS].index(nm)], zs["pred"][:C, names.index(nm)]
            parity[nm] = {"forecast_rel_maxabs": float(np.abs(a_new - a_old).max() / np.abs(a_old).max()),
                          "stored_err_mean": stored["table"][nm]["err_to_target"]["mean"],
                          "rerun_err_mean": table[nm]["err_to_target"]["mean"],
                          "shard": sinfo["arm_to_shard"][nm]}
    cp = z["corr_profile_twin"]
    res = {"question": "QA #362: does the per-token dose (twin's ||h_twin[i] - h_src[i]||) carry the twin's geometry? "
                       "Replace it with a twin-free profile from probe-split clips at the target angle (or all probe "
                       "clips); direction, token set, forward, predictor and reader unchanged",
           "n_carriers": int(C), "n_targets": int(tg.shape[1]),
           "definitions": {"doc": __doc__.split("  plan")[0].strip(),
                           "err_to_target": "predictor-native probe, step-mean forecast, mean over 4 targets per "
                                            "carrier (deg); CI = bootstrap over carriers, 1000 draws, seed 0",
                           "frac_of_uniform_to_pertoken_gap_recovered": "(err_uniform - err_twinfree) / (err_uniform "
                                                                       "- err_pertoken), ratio of carrier means, "
                                                                       "carrier bootstrap"},
           "table": table, "paired": paired, "frac_of_uniform_to_pertoken_gap_recovered": frac_gap,
           "profile_vs_twin_dose_pearson_over_set_tokens": {
               "p_bin_median": float(np.nanmedian(cp[:, 0])), "p_all_median": float(np.nanmedian(cp[:, 1]))},
           "stored_arms_copied": {nm: stored["table"][nm]["err_to_target"] for nm in
                                  ("src_chord_1x", "src_chord_uniform", "union_chord_1x")},
           "stored_references": {"unedited_err": stored["references"]["unedited_err"],
                                 "twin_ceiling_err": stored["references"]["twin_ceiling"]["err_to_target"]},
           "parity_rerun_vs_stored_sweep": parity, "forward": finfo}
    box = json.loads((root / "box.json").read_text()) if (root / "box.json").exists() else None
    smoke_info = root / "smoke_forward_info.json"
    write((TF / "smoke_results.json") if smoke else RES / "session2_disk_token_sweep_twinfree.json", res,
          stage="disk_token_sweep_twinfree_score", seeds={"bootstrap": 0},
          caches={"twinfree_pred_npz": sha256_file(root / "twinfree_pred.npz"),
                  "twinfree_forward_info_json": sha256_file(root / "twinfree_forward_info.json"),
                  "sweep_deltas_npz": sha256_file(root / "sweep_deltas.npz"),
                  "twinfree_plan_npz": sha256_file(root / "twinfree_plan.npz"),
                  "stored_sweep_json": sha256_file(RES / "session2_disk_token_sweep.json")},
          script="scripts/session2_disk_token_sweep_twinfree.py", script_sha256=sha256_file(Path(__file__)),
          forward_script_sha256=finfo["script_sha256"], box=box,
          gpu_seconds={"this_forward": finfo["seconds"],
                       "smoke": json.loads(smoke_info.read_text())["seconds"] if smoke_info.exists() else None})
    for nm, t in table.items():
        print(f"{nm:28s} err {t['err_to_target']['mean']:6.1f} {np.round(t['err_to_target']['ci95'], 1)} "
              f"R {t['R']['mean']:.3f}")
    for k, v in paired.items():
        print(k, round(v["mean"], 2), np.round(v["ci95"], 2))
    print("gap recovered", frac_gap, "parity", parity)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "plan":
        plan()
    elif cmd == "forward":
        forward(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 200)
    elif cmd == "score":
        score(sys.argv[2], smoke=len(sys.argv) > 3 and sys.argv[3] == "--smoke")

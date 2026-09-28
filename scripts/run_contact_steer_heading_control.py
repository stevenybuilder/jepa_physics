"""Contact-steer heading control (QA #367): is the injected bounce a CONTACT or just a post-contact HEADING?

scripts/run_contact_steer.py (results/p5_contact_steer.json) adds the bounce-minus-straight twin difference
(leave-one-pair-out mean over same-geometry-class donors) to the post-contact context tubelets of the 96 in-window
straight twins and reads the forecast turn toward theta_out (0.48 at point 22 all tokens, 0.45 at point 12 disk
tokens). Every control there also fits "a post-contact heading was written". This adds the deciding control on the
same carriers, same slots, same donors, same norm:

  heading difference (no wall) of pair j at point P, tubelet t:
     pool  hP[j, t] = mean over 256 tokens (straight_out_j) - same for straight_in_j
     disk  hD[j, t] = mean over straight_out_j's disk tokens - mean over straight_in_j's disk tokens
  straight_out_j = constant velocity along theta_out_j throughout, no wall, same positions as bounce_j from k_b on
  (the contact-in-context "straight-at-theta_out reference", artifacts/p5_contact_in_context/out/frames.npz).

Conditions (post-contact tubelets t with phase 1 only, exactly as run_contact_steer):
  none, a_pool_loo, b_disk_loo    rerun of the stored arms (parity + paired comparison)
  h_pool_loo      donor mean of hP (same donors as a_pool_loo), rescaled per (carrier, tubelet) to ||a_pool_loo||,
                  added to all 256 tokens
  h_disk_loo      donor mean of hD, rescaled per (carrier, tubelet) to ||b_disk_loo||, added to the carrier's disk tokens
  h_pool_loo_nat / h_disk_loo_nat  the same heading differences at their own (natural) norm
Scoring: identical forecast "transfer" reader and turn fraction as run_contact_steer.score; paired differences
(bounce arm - heading arm) per carrier with a carrier bootstrap (2000, seed 0).

  python scripts/run_contact_steer_heading_control.py plan                 (Mac)
  python scripts/run_contact_steer_heading_control.py forward --root <out> [--smoke]   (box GPU; flock /tmp/wm_gpu.lock)
  python scripts/run_contact_steer_heading_control.py score --root <out> [--smoke]     (Mac CPU)
"""
import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_contact_in_context import trajectory  # noqa: E402
from run_contact_steer import disk_token_mask, donor_mean  # noqa: E402
from run_temporal_contact import tubelet_phase, turn_fraction  # noqa: E402
from wm.data import PROJECT_ROOT  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "p5_contact_steer_heading_control"
STEER = PROJECT_ROOT / "artifacts" / "p5_contact_steer"
IN_DIR = PROJECT_ROOT / "artifacts" / "p5_contact_in_context"
OUT_DIR = PROJECT_ROOT / "artifacts" / "p5_temporal_contact"
RES = PROJECT_ROOT / "results"
LOCK = "/tmp/wm_gpu.lock"
POINTS = (22, 12)
CONDS = ("none", "a_pool_loo", "b_disk_loo", "h_pool_loo", "h_disk_loo", "h_pool_loo_nat", "h_disk_loo_nat")


# ================================================================ pure helpers (tests/test_contact_steer_heading_control.py)

def match_norm(v, ref):
    """v rescaled to ||ref|| (L2 over the last axis); zero vector stays zero."""
    v, ref = np.asarray(v, float), np.asarray(ref, float)
    nv = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(nv > 0, v / np.where(nv > 0, nv, 1) * np.linalg.norm(ref, axis=-1, keepdims=True), 0.0)


def paired(a, b, n=2000, seed=0):
    """Per-carrier paired difference a - b: mean, carrier-bootstrap 95% CI, and fraction of carriers with a > b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    m = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
            "frac_carriers_positive": float((d > 0).mean()), "n": int(len(d))}


def ratio_ci(num, den, n=2000, seed=0):
    """mean(num) / mean(den) with a carrier-bootstrap 95% CI."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = np.random.default_rng(seed).integers(0, len(num), (n, len(num)))
    r = num[idx].mean(1) / den[idx].mean(1)
    return {"ratio": float(num.mean() / den.mean()), "ci95": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]}


# ================================================================ plan (Mac)

def plan():
    cfgs = json.loads((IN_DIR / "configs.json").read_text())["configs"]
    mO = np.stack([disk_token_mask(trajectory(c, "straight_out")) for c in cfgs])
    Pl = dict(np.load(STEER / "plan.npz"))
    ART.mkdir(parents=True, exist_ok=True)
    np.savez(ART / "plan.npz", mask_straight=Pl["mask_straight"], mask_bounce=Pl["mask_bounce"], mask_out=mO,
             kb=Pl["kb"])
    print("disk tokens per tubelet: straight", Pl["mask_straight"].sum(-1).mean(), "straight_out", mO.sum(-1).mean())


# ================================================================ forward (box GPU)

def forward(root, batch=8, smoke=False):
    import torch
    from wm.extract import load_model, pick_device, preprocess, set_precision
    from wm.predictor_readout import pool_steps, predict_future
    from wm.propagate import suffix

    set_precision()
    dev = pick_device()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    FR = np.load(IN_DIR / "out" / "frames.npz")
    Pl = dict(np.load(ART / "plan.npz"))
    same = json.loads((STEER / "donors.json").read_text())["same_class_loo"]
    kb = Pl["kb"]
    n = len(kb)
    phase = np.stack([tubelet_phase(k) for k in kb])[:, :4]
    carriers = np.arange(4) if smoke else np.arange(n)
    ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16)   # noqa: E731
    out, info = {}, {"norms": {}}

    def enc_to(model, px, point):
        enc = model.encoder
        h = enc.embeddings(px[:, :8]).float()
        for layer in enc.layer[:point]:
            h = layer(h, None, None, False)[0]
        return h

    def dmean(h, m):
        w = m.float()
        return (h * w[..., None]).sum(2) / w.sum(2).clamp(min=1)[..., None]

    with open(LOCK, "a") as lk:
        tw = time.time()
        fcntl.flock(lk, fcntl.LOCK_EX)
        waited = time.time() - tw
        print(f"lock after {waited:.0f}s", flush=True)
        t0 = time.time()
        with torch.no_grad():
            model = load_model("vjepa2", dev)
            for P in POINTS:
                hs = {}
                for kind in ("straight_in", "bounce", "straight_out"):
                    F = FR[kind]
                    parts = []
                    for s in range(0, n, batch):
                        with ac():
                            parts.append(enc_to(model, preprocess(list(F[s:s + batch])).to(dev), P).cpu())
                    hs[kind] = torch.cat(parts).view(n, 4, 256, -1)
                hS, hB, hO = hs["straight_in"], hs["bounce"], hs["straight_out"]
                mS, mB, mO = (torch.as_tensor(Pl[k]).view(n, 4, 256) for k in ("mask_straight", "mask_bounce", "mask_out"))
                dP = (hB.mean(2) - hS.mean(2)).numpy()
                dD = (dmean(hB, mB) - dmean(hS, mS)).numpy()
                hP = (hO.mean(2) - hS.mean(2)).numpy()
                hD = (dmean(hO, mO) - dmean(hS, mS)).numpy()
                for k, v in (("dP", dP), ("dD", dD), ("hP", hP), ("hD", hD)):
                    out[f"P{P}/{k}"] = v.astype(np.float32)
                res = {c: [] for c in CONDS}
                norms = {c: [] for c in CONDS}
                cosines = {"pool": [], "disk": []}
                for s in range(0, len(carriers), batch):
                    ids = carriers[s:s + batch]
                    base = hS[ids].clone()
                    for c in CONDS:
                        h = base.clone()
                        for bi, i in enumerate(ids):
                            for t in range(4):
                                if phase[i, t] != 1 or c == "none":
                                    continue
                                kind = "disk" if "disk" in c else "pool"
                                bref = donor_mean(dD if kind == "disk" else dP, phase, same[i], t)
                                if c.startswith("h_"):
                                    v = donor_mean(hD if kind == "disk" else hP, phase, same[i], t)
                                    if c == "h_pool_loo":
                                        cosines["pool"].append(float(v @ bref / np.linalg.norm(v) / np.linalg.norm(bref)))
                                    if c == "h_disk_loo":
                                        cosines["disk"].append(float(v @ bref / np.linalg.norm(v) / np.linalg.norm(bref)))
                                    if not c.endswith("_nat"):
                                        v = match_norm(v, bref)
                                else:
                                    v = bref
                                add = torch.as_tensor(v, dtype=torch.float32)
                                norms[c].append(float(add.norm()))
                                if kind == "disk":
                                    h[bi, t, mS[i, t]] += add
                                else:
                                    h[bi, t] += add
                        h = h.view(len(ids), 1024, -1).to(dev)
                        with ac():
                            r = suffix(model, h, P, None, return_final=True)
                            fc = pool_steps(predict_future(model, r["final"])).float().cpu().numpy()
                        res[c].append(fc)
                    print(f"P{P} carriers {s + len(ids)}/{len(carriers)} {time.time() - t0:.0f}s", flush=True)
                for c in CONDS:
                    out[f"P{P}/{c}/forecast"] = np.concatenate(res[c])
                info["norms"][str(P)] = {c: (float(np.mean(v)) if v else 0.0) for c, v in norms.items()}
                info[f"cos_heading_vs_bounce_P{P}"] = {k: float(np.median(v)) for k, v in cosines.items()}
                info["token_norm_mean_P" + str(P)] = float(hS.norm(dim=-1).mean())
        secs = time.time() - t0
    info.update({"seconds_gpu_total": secs, "lock_wait_s": waited, "gpu": torch.cuda.get_device_name(0),
                 "torch": torch.__version__, "smoke": smoke, "n_carriers": int(len(carriers)),
                 "precision": "context encode to P, suffix and predictor bf16 autocast, fp32 residual/edits"})
    np.savez(root / "forward.npz", carriers=carriers, **out)
    (root / "forward_info.json").write_text(json.dumps(info, indent=1))
    print("done", secs, flush=True)


# ================================================================ score (Mac CPU)

def forecast_readers():
    """The 'transfer' forecast reader of run_contact_steer.score (identical construction)."""
    from temporal_contact_score import fit_probe
    from wm.data import load_table
    from wm.p2_data import load_inputs
    from wm.probes import targets
    FO = dict(np.load(OUT_DIR / "out" / "forward.npz"))
    co = json.loads((OUT_DIR / "bounces.json").read_text())["configs"]
    kbo = np.array([c["k_b"] for c in co])
    tio, too = np.array([c["theta_in"] for c in co]), np.array([c["theta_out"] for c in co])
    pho = np.stack([tubelet_phase(k) for k in kbo])
    df = load_table("direction")
    Ydir = targets(df, "direction")[0]
    prow = np.where(load_inputs("direction", 0)["role"] == "probe")[0]
    Zall = np.load(PROJECT_ROOT / "artifacts/session2/native/pred_pooled_all.npy").astype(np.float64)

    def sc(th):
        t = np.radians(th)
        return np.stack([np.sin(t), np.cos(t)], -1)
    readers = []
    for s in range(4):
        t = 4 + s
        m = pho[:, t] != 0
        X = np.concatenate([Zall[prow][:, s], FO["ct/straight/forecast"][:, s], FO["ct/bounce/forecast"][m, s]])
        Y = np.concatenate([Ydir[prow], sc(tio), sc(np.where(pho[m, t] == 1, too[m], tio[m]))])
        readers.append(fit_probe(X, Y, "circular"))
    return readers


def score(root, smoke=False):
    from run_session2 import write
    from run_token_patching import summarize
    from temporal_contact_score import ang, apply
    from wm.provenance import sha256_file

    root = Path(root)
    F = dict(np.load(root / "forward.npz"))
    finfo = json.loads((root / "forward_info.json").read_text())
    car = F["carriers"]
    cfgs = json.loads((IN_DIR / "configs.json").read_text())["configs"]
    th_in = np.array([c["theta_in"] for c in cfgs])[car]
    th_out = np.array([c["theta_out"] for c in cfgs])[car]
    FI = dict(np.load(IN_DIR / "out" / "forward.npz"))
    readers = forecast_readers()

    def tf(Fc):
        return np.stack([turn_fraction(ang(apply(readers[s][0], Fc[:, s])), th_in, th_out) for s in range(4)], 1).mean(1)

    stored = json.loads((RES / "p5_contact_steer.json").read_text())
    Fst = dict(np.load(STEER / "out" / "forward.npz"))
    R = {"references": {"real_bounce_clips_forecast_step_mean": summarize(tf(FI["bounce/forecast"][car])),
                        "straight_out_clips_forecast_step_mean": summarize(tf(FI["straight_out/forecast"][car])),
                        "unedited_straight_twins_forecast_step_mean": summarize(tf(FI["straight_in/forecast"][car]))},
         "points": {}}
    for P in POINTS:
        m = {c: tf(F[f"P{P}/{c}/forecast"]) for c in CONDS}
        blk = {"delta_norm_mean": finfo["norms"][str(P)], "token_norm_mean": finfo[f"token_norm_mean_P{P}"],
               "cos_median_heading_vs_bounce_donor_mean": finfo[f"cos_heading_vs_bounce_P{P}"],
               "forecast_turn_fraction_step_mean": {c: summarize(v) for c, v in m.items()}}
        st = {c: tf(Fst[f"P{P}/{c}/forecast"][car]) for c in ("none", "a_pool_loo", "b_disk_loo")}
        blk["parity_rerun_vs_stored_forward"] = {
            c: {"forecast_rel_maxabs": float(np.abs(F[f"P{P}/{c}/forecast"] - Fst[f"P{P}/{c}/forecast"][car]).max()
                                             / np.abs(Fst[f"P{P}/{c}/forecast"][car]).max()),
                "turn_fraction_maxabs_diff": float(np.abs(m[c] - st[c]).max())} for c in st}
        blk["stored_p5_contact_steer_json"] = {
            c: stored["results"]["points"][str(P)]["conditions"][c]["forecast_turn_fraction_step_mean"]["all"]
            for c in ("none", "a_pool_loo", "b_disk_loo", "c_rand_pool", "c_rand_disk", "d_far_pool", "d_far_disk",
                      "e_own_pool", "e_own_disk", "e_own_tokens")}
        blk["paired"] = {
            "a_pool_loo_minus_h_pool_loo": paired(m["a_pool_loo"], m["h_pool_loo"]),
            "b_disk_loo_minus_h_disk_loo": paired(m["b_disk_loo"], m["h_disk_loo"]),
            "a_pool_loo_minus_h_pool_loo_nat": paired(m["a_pool_loo"], m["h_pool_loo_nat"]),
            "b_disk_loo_minus_h_disk_loo_nat": paired(m["b_disk_loo"], m["h_disk_loo_nat"]),
            "h_pool_loo_minus_none": paired(m["h_pool_loo"], m["none"]),
            "h_disk_loo_minus_none": paired(m["h_disk_loo"], m["none"]),
            "a_pool_loo_minus_none": paired(m["a_pool_loo"], m["none"]),
            "b_disk_loo_minus_none": paired(m["b_disk_loo"], m["none"])}
        blk["ratio_heading_over_bounce_edit_effect"] = {
            "pool": ratio_ci(m["h_pool_loo"] - m["none"], m["a_pool_loo"] - m["none"]),
            "disk": ratio_ci(m["h_disk_loo"] - m["none"], m["b_disk_loo"] - m["none"])}
        R["points"][str(P)] = blk
    site = {"22": ("a_pool_loo", "h_pool_loo"), "12": ("b_disk_loo", "h_disk_loo")}
    headline = {P: {"site": "block 22 all tokens" if P == "22" else "block 12 disk tokens",
                    "bounce_edit": R["points"][P]["forecast_turn_fraction_step_mean"][site[P][0]],
                    "no_wall_heading_edit_norm_matched": R["points"][P]["forecast_turn_fraction_step_mean"][site[P][1]],
                    "paired_bounce_minus_heading": R["points"][P]["paired"][f"{site[P][0]}_minus_{site[P][1]}"]}
                for P in site}
    res = {"question": "QA #367: does the contact steer write a contact, or only a post-contact heading? Same carriers, "
                       "slots, donors and norm as the bounce edit; heading difference with no wall "
                       "(straight_out - straight_in)",
           "headline": headline,
           "definitions": {"doc": __doc__.split("  python")[0].strip(),
                           "turn_fraction": "0 = carrier's theta_in, 1 = its twin's theta_out; step mean over the 4 "
                                            "forecast steps",
                           "forecast_reader": "identical to run_contact_steer.score ('transfer' reader: native probe-"
                                              "role forecasts + p5_contact_dynamics rendered forecasts)",
                           "ci95": "carrier bootstrap, 2000 resamples, seed 0 (summarize / paired)"},
           "results": R, "config": {"points": POINTS, "conditions": CONDS, "forward": finfo}}
    out = (ART / "smoke_results.json") if smoke else RES / "p5_contact_steer_heading_control.json"
    fsha = root / "forward_script_sha256.txt"
    write(out, res, seeds={"bootstrap": 0, "random": "none (no random arms)"},
          activation_sha256={"forward_npz": sha256_file(root / "forward.npz"),
                             "forward_info_json": sha256_file(root / "forward_info.json"),
                             "plan_npz": sha256_file(ART / "plan.npz"), "donors_json": sha256_file(STEER / "donors.json"),
                             "frames_npz": sha256_file(IN_DIR / "out" / "frames.npz"),
                             "configs_json": sha256_file(IN_DIR / "configs.json"),
                             "in_context_forward_npz": sha256_file(IN_DIR / "out" / "forward.npz"),
                             "stored_contact_steer_forward_npz": sha256_file(STEER / "out" / "forward.npz"),
                             "stored_p5_contact_steer_json": sha256_file(RES / "p5_contact_steer.json")},
          script="scripts/run_contact_steer_heading_control.py", script_sha256=sha256_file(Path(__file__)),
          forward_script_sha256=fsha.read_text().split()[0] if fsha.exists() else None,
          box="Vast 53255819 (box 4, RTX 4060 Ti)",
          gpu_seconds={"this_forward": finfo["seconds_gpu_total"],
                       "smoke": (json.loads((root / "smoke_forward_info.json").read_text())["seconds_gpu_total"]
                                 if (root / "smoke_forward_info.json").exists() else None)},
          n_carriers=int(len(car)))
    for P in site:
        h = headline[P]
        print(P, "bounce", round(h["bounce_edit"]["mean"], 3), "heading", round(h["no_wall_heading_edit_norm_matched"]["mean"], 3),
              "diff", h["paired_bounce_minus_heading"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "forward", "score"])
    ap.add_argument("--root", default=str(ART / "out"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        plan()
    elif a.cmd == "forward":
        forward(a.root, smoke=a.smoke)
    else:
        score(a.root, smoke=a.smoke)


if __name__ == "__main__":
    main()

"""Attentive-MLP probes (paper §3.2; ledger #221 / #246) on V-JEPA 2 per-patch tokens, beside the mean-pool ridge.

Tokens: one 1024-vector per spatial position (16 x 16 = 256 per clip), averaged over the 8 token time steps, from the
p1a_perpatch.py extract stage (direction: /workspace/wm/artifacts/perpatch/direction_vjepa2.npy, read in place;
speed / acceleration: extracted at the three points below into /dev/shm/wm_vmae). Probe: wm.attentive (learned
queries, softmax over the 256 tokens, LayerNorm + MLP head), Adam, fixed budget; 5-fold CV over the split's stored
train folds (splits/split_v1.json, the folds the mean-pool ridge's cv_mean used), test rows never touched in CV; then
one refit on all train rows scored on the test rows. Points per variable = REPORT §3.1 onset / paper layer / peak.

  python scripts/run_attentive_probe.py --variable direction --memmap /workspace/wm/artifacts/perpatch/direction_vjepa2.npy
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import PROJECT_ROOT, load_table  # noqa: E402

POINTS = {"direction": {"onset": 2, "paper_layer": 9, "peak": 22},
          "speed": {"onset": 1, "paper_layer": 9, "peak": 19},
          "acceleration": {"onset": 1, "paper_layer": 9, "peak": 21}}
HP = dict(heads=4, hidden=128, dropout=0.1, epochs=60, batch=64, seed=0)
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 5e-3]        # App. B learning-rate grid (refs/physics_paper.txt l.692)
WDS = [0.01, 0.1]                            # first two of App. B's weight-decay grid {0.01, 0.1, 0.4, 0.8} (time budget)
CONFIGS = [{**HP, "lr": lr, "weight_decay": wd} for lr in LRS for wd in WDS]


def main():
    import torch
    from wm.attentive import cv_sweep, fit_predict
    from wm.probes import split_rows, targets
    from wm.provenance import sha256_file

    ap = argparse.ArgumentParser()
    ap.add_argument("--variable", required=True, choices=tuple(POINTS))
    ap.add_argument("--memmap", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--mac-results", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    var = a.variable
    side = json.loads(Path(a.memmap).with_suffix(".json").read_text())
    assert side["complete"]
    mm = np.load(a.memmap, mmap_mode="r")
    df = load_table(var)
    assert [int(i) for i in df["id"]] == side["ids"]
    Y, kind, score_fn = targets(df, var)
    split_file = PROJECT_ROOT / "splits" / "split_v1.json"
    tr, te, folds = split_rows(var, df, split_file)
    mp = json.loads((Path(a.mac_results) / f"p1a_{var}_{var}_meanpool.json").read_text())
    mp_by = {r["point"]: r for r in mp["layers"]}
    rows = []
    t_all = time.time()
    for role, point in POINTS[var].items():
        t0 = time.time()
        pi = side["points"].index(point)
        X = np.asarray(mm[:, pi], np.float32)                                      # [N, 256, 1024], in place read
        sweep = cv_sweep(X[tr], Y[tr], folds, score_fn, CONFIGS, device=a.device)
        cv = max(sweep, key=lambda r: r["cv_r2_mean"])                         # chosen by CV mean, like ridge alpha
        best = cv["config"]
        (Pte,) = fit_predict(X[tr], Y[tr], [X[te]], device=a.device, **best)
        st = score_fn(Y[te], Pte)
        r = mp_by[point]
        row = {"role": role, "point": point, "frac": min(point, 24) / 24,
               "attentive": {"best_config": {"lr": best["lr"], "weight_decay": best["weight_decay"]},
                             "sweep": [{"lr": r["config"]["lr"], "weight_decay": r["config"]["weight_decay"],
                                        "cv_r2_mean": r["cv_r2_mean"], "cv_r2_sd": r["cv_r2_sd"],
                                        "cv_mae_mean": r["cv_mae_mean"]} for r in sweep],
                             "cv_r2_mean": cv["cv_r2_mean"], "cv_r2_sd": cv["cv_r2_sd"], "cv_mae_mean": cv["cv_mae_mean"],
                             "cv_mae_sd": cv["cv_mae_sd"], "folds": cv["folds"],
                             "test_r2": float(st["r2"]), "test_mae": float(st["mae"])},
               "meanpool_ridge": {"file": f"results/p1a_{var}_{var}_meanpool.json", "key": f"layers[point={point}]",
                                  "cv_mean": r["cv_mean"], "cv_sd": r["cv_sd"], "cv_mae_mean": r["cv_mae_mean"],
                                  "test_r2": r["test_r2"], "test_mae": r["test_mae"], "alpha": r["alpha"]},
               "attentive_minus_meanpool_cv_r2": cv["cv_r2_mean"] - r["cv_mean"],
               "seconds": time.time() - t0}
        rows.append(row)
        print(f"{var} {role} pt {point}: attentive CV R2 {cv['cv_r2_mean']:.4f} ± {cv['cv_r2_sd']:.4f} MAE {cv['cv_mae_mean']:.4f} "
              f"| test {st['r2']:.4f} | meanpool ridge CV {r['cv_mean']:.4f} test {r['test_r2']:.4f} ({time.time() - t0:.0f}s)",
              flush=True)
        del X
    out = {"status": "EXTENSION: the paper's attentive-MLP probe results are on IntPhys only (§3.2); this applies an attentive probe to the supplied probing sets",
           "caveat_time_averaged_tokens": ("tokens are the 256 spatial positions each AVERAGED over the 8 token time steps, not the "
                                           "2,048 space-time tokens; §3.2's stated reason for attentive probes is that pooling "
                                           "'can obscure spatial or temporal structure', so this probe sees spatial but not temporal structure"),
           "variable": var, "kind": kind, "target": "(sin, cos)" if var == "direction" else f"{var} magnitude",
           "units_mae": "degrees (circular)" if var == "direction" else ("m/s" if var == "speed" else "m/s^2"),
           "model": "vjepa2", "n_train": int(len(tr)), "n_test": int(len(te)), "points": POINTS[var], "layers": rows,
           "methods": {
               "tokens": "per spatial position (16x16 = 256 tokens per clip), each averaged over the 8 token time steps (p1a_perpatch.py extract; fp16 storage)",
               "probe": ("wm.attentive.AttentiveProbe: 4 learned queries (one per head) applied directly to train-standardised tokens, "
                         "softmax over the 256 tokens, concatenated per-head weighted means -> LayerNorm -> Dropout 0.1 -> "
                         "Linear(4096,128) -> GELU -> Linear(128, out); not identical to a linear-K/V attention block because of "
                         "the LayerNorm (module docstring)"),
               "training": f"AdamW, {HP['epochs']} epochs, batch {HP['batch']}, MSE on standardised targets, seed {HP['seed']}, no early stopping; lr x weight_decay swept over {LRS} x {WDS} (App. B lr grid; first two App. B weight decays), chosen by 5-fold CV mean R2 on the train folds (the reported cv_r2_mean is the max over the sweep, as the ridge's cv_mean is the max over alphas); heads 4, hidden 128, dropout 0.1 fixed",
               "cv": "5-fold over the split's stored train folds (splits/split_v1.json `fold`, stratified by value; the same folds as the mean-pool ridge cv_mean); standardisation re-fit inside each fold; test rows unused",
               "test": "one refit on all train rows, test rows scored once",
               "score": "wm.probes.score: R2 (mean over target columns), MAE (circular degrees for direction)",
               "points": "REPORT §3.1: mean-pool onset / paper emergence-zone layer (point 9 = paper layer 8) / mean-pool peak",
               "comparison": "meanpool_ridge = results/p1a_{var}_{var}_meanpool.json layers[point]: closed-form ridge on the fp32 token mean, alpha by the same folds"},
           "provenance": {"commit": a.commit, "memmap": a.memmap, "memmap_points": side["points"],
                          "memmap_parity": side.get("meanpool_parity"), "split_file": "splits/split_v1.json",
                          "split_sha256": sha256_file(split_file), "device": a.device, "torch_threads": a.threads,
                          "host": "vast 53030966 (box 1)", "wall_seconds": time.time() - t_all,
                          "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    path = Path(a.results_dir) / f"p1a_attentive_{var}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
